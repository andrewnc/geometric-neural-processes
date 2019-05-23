import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# from scipy.stats import wasserstein_distance
# from geomloss import SamplesLoss
from utils import sliced_wasserstein_distance
from tqdm import tqdm


import os

import utils

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
    

class Encoder(nn.Module):
    """takes in context points and returns a fixed length aggregation"""
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_tile = nn.Conv2d(3,16, kernel_size=1)
        self.fc_tile = nn.Linear(256, 256)
        self.fc1 = nn.Linear(256+16, 128)
        self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        

    def forward(self, x, tile):
        """x = coord of upper left of tile
        this returns the aggregated r value
        """
        tile = tile.transpose(3,2).transpose(2,1)
        tile = self.conv_tile(tile)
        y = self.fc_tile(tile.view(tile.shape[0], tile.shape[1]*tile.shape[2]*tile.shape[3]))
        x = torch.stack(([coords.flatten() for coords in x]))
        x = torch.cat((x,y), -1)
        # original_x = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        output = self.fc4(x)
        return output.mean(0).view(1, 128)

class Decoder(nn.Module):
    def __init__(self, m, n):
        super(Decoder, self).__init__()
        self.m = m
        self.n = n
        self.fc1 = nn.Linear(144, 144)
        self.fc2 = nn.Linear(144, 144)
        self.fc3 = nn.Linear(144, 144)
        self.fc4 = nn.Linear(144, 48)

    def forward(self, r, inds):
        """r is the aggregated data used to condition"""
        #x = torch.tensor([[i, j] for i in range(0,self.m) for j in range(0,self.n)]).float().to(device)
        #out = torch.cat((x, r.view(1,-1).repeat(1,self.m*self.n).view(self.m*self.n,128)), 1)
        inds = torch.stack(([coords.flatten() for coords in inds]))
        x = torch.cat((inds, r.repeat(inds.shape[0], 1)), -1)
        # original_x = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = self.fc2(x) + original_x
        x = F.relu(self.fc3(x))
        out = torch.sigmoid(self.fc4(x))
        return out.view(64,4,4,3)


# we probably want a better critic
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*32, 28*28) 
        self.fc2 = nn.Linear(28*28, 20*20) 
        self.fc3 =nn.Linear(20*20, 16*16) 
        self.fc4 =nn.Linear(16*16, 10*10) 
        self.output = nn.Linear(10*10, 1)

    def forward(self, x):
        x = x.contiguous().view(-1, 32*32)
        out = self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))))
        out = self.output(out)
        return out.mean()

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()
if __name__ == "__main__":

    batch_size=64
    test_batch_size=1

    c, m, n = 3, 32,32
    single_class = 0


    kwargs = {'num_workers': 10, 'pin_memory': True} if use_cuda else {}
    # kwargs = {}

    train_loader = torch.utils.data.DataLoader(
        datasets.cifar.CIFAR10('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.0,), (1.0,)),
                    ])),
        batch_size=batch_size, shuffle=True, **kwargs)


    test_loader = torch.utils.data.DataLoader(
        datasets.cifar.CIFAR10('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.0,), (1.0,)),
                    ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    epochs = 10
    log_interval = 50

    encoder = Encoder().to(device)
    decoder = Decoder(m, n).to(device)
    # critic = Critic().to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    # criterion = nn.MSELoss()
    # criterion = SamplesLoss("sinkhorn", p=2, blur=1.)
    # optimizer_critic = optim.RMSprop(critic.parameters(), lr=0.1)
    loss_values = []
    
    for epoch in range(1, epochs+1):
        encoder.train()
        decoder.train()
        # critic.train()

        progress = tqdm(train_loader)
        # progress = tqdm(range(100))
        # for _ in progress: 
        for (ground_truth_images, targets) in progress:
            for image, target in zip(ground_truth_images, targets):
                if target != single_class:
                    continue
                optimizer.zero_grad()
                num_tiles = np.random.randint(4, 64)
                tiles, inds, _, all_inds, tiled_ground_truth = utils.get_tiles(image, num_tiles=num_tiles)

                # tiles = tiled_ground_truth[26].repeat(len(tiles),1,1,1)
                # weird_ground_truth = tiled_ground_truth[26].repeat(len(tiled_ground_truth),1,1,1)
                weird_ground_truth = tiled_ground_truth
                image = image.to(device)
                tiles = tiles.float().to(device)
                tiled_ground_truth = tiled_ground_truth.float().to(device)
                weird_ground_truth = weird_ground_truth.float().to(device)
                one_hot_identity = torch.eye(8) # avoid magic numbers
                inds = inds / 4
                all_inds = all_inds / 4

                one_hot = one_hot_identity[inds].float().to(device)

                r = encoder(one_hot, tiles)
                fake_image = decoder(r, one_hot_identity[all_inds].float().to(device))
                # loss = torch.mean(torch.tensor([wasserstein_distance(x, y) for (x,y) in zip(fake_image.view(64,3*4*4).detach().cpu(), tiled_ground_truth.view(64,3*4*4).detach().cpu())], requires_grad=True))


                # loss = criterion(fake_image, weird_ground_truth)
                fake_image = fake_image.view(64,4*4*3)
                weird_ground_truth = weird_ground_truth.view(64,4*4*3)
                loss = 0.0
                for tile in range(64):
                    loss += sliced_wasserstein_distance(fake_image[tile].unsqueeze(0), weird_ground_truth[tile].unsqueeze(0),num_projections=5, device=device)
                            
                loss.backward()
                optimizer.step()
            # plot_grad_flow(encoder.named_parameters())
            # plot_grad_flow(decoder.named_parameters())


        # for t in range(5):
            # optimizer_critic.zero_grad()
            # for image, target in zip(ground_truth_images, targets):
            #     num_tiles = np.random.randint(4, 64)
            #     tiles, inds, _, all_inds, tiled_ground_truth = utils.get_tiles(image, num_tiles=num_tiles)

            #     image = image.to(device)
            #     tiles = tiles.float().to(device)
            #     inds = inds.float().to(device)
            #     all_inds = all_inds.float().to(device)
            #     tiled_ground_truth = tiled_ground_truth.float().to(device)

            #     # run the model to get r which will be concatenated onto every node pair in the decoder
            #     r = encoder(inds, tiles)

            #     fake_image = decoder(r, all_inds)

        #         disc_real = critic(tiled_ground_truth) # we made need to change this to tiles?
        #         disc_fake = critic(fake_image)
        #         # gradient_penalty = utils.calc_gradient_penalty(critic, tiled_ground_truth, fake_image)
        #         loss = disc_fake - disc_real# + gradient_penalty
        #         loss.backward()
        #         # w_dist = disc_real - disc_fake
        #     optimizer_critic.step()
        #     for p in critic.parameters():
        #         p.data.clamp_(-0.01, 0.01)


        # optimizer.zero_grad()
        # for image, target in zip(ground_truth_images, targets):
        #     num_tiles = np.random.randint(4, 64)
        #     tiles, inds, _, all_inds, tiled_ground_truth = utils.get_tiles(image, num_tiles=num_tiles)

        #     image = image.to(device)
        #     tiles = tiles.float().to(device)
        #     inds = inds.float().to(device)
        #     all_inds = all_inds.float().to(device)
        #     tiled_ground_truth = tiled_ground_truth.float().to(device)

        #     # run the model to get r which will be concatenated onto every node pair in the decoder
        #     r = encoder(inds, tiles)

        #     fake_image = decoder(r, all_inds)
        #     disc_fake = critic(fake_image)
        #     # disc_fake.backward()
        #     gen_loss = - disc_fake
        #     gen_loss.backward()
            # progress.set_description("E{} - L{:.4f}".format(epoch, loss.item()))
            progress.set_description("E{} - L{:.4f} - EMin{:.4f} - EMax{:.4f} - DMin{:.4f} - DMax{:.4f}".format(epoch, loss.item(),
            torch.min(torch.tensor([torch.min(p.grad) for p in encoder.parameters()])), 
            torch.max(torch.tensor([torch.max(p.grad) for p in encoder.parameters()])), 
            torch.min(torch.tensor([torch.min(p.grad) for p in decoder.parameters()])), 
            torch.max(torch.tensor([torch.max(p.grad) for p in decoder.parameters()])) 
            ))
            loss_values.append(loss.item())

            if loss.item() <= 0.0001:
                break
        with open("encoder.pkl", "wb") as of:
            pickle.dump(encoder, of)

        with open("decoder.pkl", "wb") as of:
            pickle.dump(decoder, of)

        with open("loss.pkl", "wb") as of:
            pickle.dump(loss_values, of)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():

            for i, (ground_truth_image, target) in enumerate(train_loader):
                ground_truth_image = ground_truth_image[0].view(3, 32, 32)

                if target[0] != single_class:
                    continue

                tiles, inds, frame, all_inds, tiled_ground_truth = utils.get_tiles(ground_truth_image, num_tiles=32)


                # tiles = tiled_ground_truth[26].repeat(len(tiles),1,1,1)
                # weird_ground_truth = tiled_ground_truth[26].repeat(len(tiled_ground_truth),1,1,1)
                weird_ground_truth = tiled_ground_truth
                image = image.to(device)
                tiles = tiles.float().to(device)
                tiled_ground_truth = tiled_ground_truth.float().to(device)
                weird_ground_truth = weird_ground_truth.float().to(device)
                one_hot_identity = torch.eye(8) # avoid magic numbers
                _inds = inds /4
                _all_inds = all_inds /4

                one_hot = one_hot_identity[_inds].float().to(device)

                r = encoder(one_hot, tiles)
                generated_image = decoder(r, one_hot_identity[_all_inds].float().to(device))

                for j, tile in enumerate(tiled_ground_truth):
                    for ind in inds.detach().cpu():
                        if all_inds[j][0].cpu().long() == ind[0].long() and all_inds[j][1].cpu().long() == ind[1].long():
                            frame[j] = tiled_ground_truth[j]

                fig, ax = plt.subplots(nrows=8, ncols=8)
                count = 0
                for row in range(8):
                    for col in range(8):
                        ax[row][col].axis("off")
                        ax[row][col].imshow(frame[count].detach().cpu())
                        count += 1
                plt.axis("off")
                plt.savefig("{}sparse{}.png".format(epoch, i))
                plt.close()


                fig, ax = plt.subplots(nrows=8, ncols=8)
                count = 0
                for row in range(8):
                    for col in range(8):
                        ax[row][col].axis("off")
                        ax[row][col].imshow(generated_image[count].detach().cpu())
                        count += 1
                plt.axis("off")
                plt.savefig("{}fake{}.png".format(epoch, i))
                plt.close()


                fig, ax = plt.subplots(nrows=8, ncols=8)
                count = 0
                for row in range(8):
                    for col in range(8):
                        ax[row][col].axis("off")
                        ax[row][col].imshow(weird_ground_truth[count].detach().cpu())
                        count += 1
                plt.axis("off")
                plt.savefig("{}groundtruth{}.png".format(epoch,i))
                plt.close()

                if i >= 20:
                    break
