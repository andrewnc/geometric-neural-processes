import matplotlib.pyplot as plt
import numpy as np

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


from tqdm import tqdm

import os

import utils

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
    

class Encoder(nn.Module):
    """takes in context points and returns a fixed length aggregation"""
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_tile = nn.Conv2d(3,2, kernel_size=1)
        self.fc_tile = nn.Linear(4*4, 1)
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 128)
        

    def forward(self, x, tile):
        """x = coord of upper left of tile
        this returns the aggregated r value
        """
        tile = tile.transpose(3,2).transpose(2,1)
        tile = self.conv_tile(tile)
        y = self.fc_tile(tile.view(tile.shape[0], tile.shape[1], tile.shape[2]*tile.shape[3]))
        y = y.squeeze(-1)
        x = torch.cat((x,y), -1)
        output = self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))))

        return output.mean(0).view(1, 128)

class Decoder(nn.Module):
    def __init__(self, m, n):
        super(Decoder, self).__init__()
        self.m = m
        self.n = n
        self.fc1 = nn.Linear(130, 86)
        self.fc2 = nn.Linear(86, 48)
        # self.r_conv1 = nn.Conv2d(3,3,kernel_size=3, stride=3)
        # self.r_conv2 = nn.Conv2d(3,3,kernel_size=3, stride=3)
        # self.r_conv3 = nn.Conv2d(3,3,kernel_size=3, stride=3)
        # self.conv1 = nn.ConvTranspose2d(65, 32,kernel_size=5, stride=2)
        # self.conv2 = nn.ConvTranspose2d(32, 16,kernel_size=5, stride=2)
        # self.conv3 = nn.ConvTranspose2d(16, 8,kernel_size=5, stride=1)
        # self.conv4 = nn.ConvTranspose2d(8, 1,kernel_size=4, stride=1)

    def forward(self, r, inds):
        """r is the aggregated data used to condition"""
        #x = torch.tensor([[i, j] for i in range(0,self.m) for j in range(0,self.n)]).float().to(device)
        #out = torch.cat((x, r.view(1,-1).repeat(1,self.m*self.n).view(self.m*self.n,128)), 1)
        x = torch.cat((inds, r.repeat(inds.shape[0], 1)), -1)
        return torch.sigmoid(self.fc2(F.relu(self.fc1(x)))).view(64,3,4,4)
        # r = self.r_conv3(F.relu(self.r_conv2(F.relu(self.r_conv1(r)))))
        # out = torch.cat((r, frame), 0)
        # out = out.transpose(0,1)
        # h = self.conv4(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(out)))))))
        # return torch.sigmoid(h.squeeze(1))

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


if __name__ == "__main__":

    batch_size=64
    test_batch_size=1

    c, m, n = 3, 32,32
    num_pixels = m*n

    min_context_points = num_pixels * 0.15 # always have at least 15% of all pixels
    max_context_points = num_pixels * 0.95 # always have at most 95% of all pixels

    # normalizer = transforms.Normalize((0.1307,), (0.3081,))

    # train_loader = normalizer(torch.load("../data/processed/training.pt")[0].float())
    # test_loader = normalizer(torch.load("../data/processed/test.pt")[0].float())


    kwargs = {'num_workers': 1, 'pin_memory': False} if use_cuda else {}
    # kwargs = {}

    train_loader = torch.utils.data.DataLoader(
        datasets.cifar.CIFAR10('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        # transforms.Normalize((0.1307,), (0.3081,)),
                    ])),
        batch_size=batch_size, shuffle=True, **kwargs)


    test_loader = torch.utils.data.DataLoader(
        datasets.cifar.CIFAR10('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        # transforms.Normalize((0.1307,), (0.3081,)),
                    ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    epochs = 10
    log_interval = 50

    encoder = Encoder().to(device)
    decoder = Decoder(m, n).to(device)
    critic = Critic().to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))
    optimizer_critic = optim.Adam(critic.parameters())
    
    for epoch in range(1, epochs+1):
        encoder.train()
        decoder.train()

        progress = tqdm(train_loader)

        total_loss = 0
        count = 0
        for (ground_truth_images, targets) in progress:
            optimizer.zero_grad()
            for image, target in zip(ground_truth_images, targets):
                num_tiles = np.random.randint(4, 64)
                tiles, inds, _, all_inds, tiled_ground_truth = utils.get_tiles(image, num_tiles=num_tiles)

                image = image.to(device)
                tiles = tiles.float().to(device)
                inds = inds.float().to(device)
                all_inds = all_inds.float().to(device)
                tiled_ground_truth = tiled_ground_truth.float().to(device)

                # run the model to get r which will be concatenated onto every node pair in the decoder
                r = encoder(inds, tiles)

                fake_image = decoder(r, all_inds)
                disc_fake = critic(fake_image)
                disc_fake.backward()
                gen_loss = - disc_fake
            optimizer.step()

            for t in range(5):
                optimizer_critic.zero_grad()
                loss = 0
                for image, target in zip(ground_truth_images, targets):
                    num_tiles = np.random.randint(4, 64)
                    tiles, inds, _, all_inds, tiled_ground_truth = utils.get_tiles(image, num_tiles=num_tiles)

                    image = image.to(device)
                    tiles = tiles.float().to(device)
                    inds = inds.float().to(device)
                    all_inds = all_inds.float().to(device)
                    tiled_ground_truth = tiled_ground_truth.float().to(device)

                    # run the model to get r which will be concatenated onto every node pair in the decoder
                    r = encoder(inds, tiles)

                    fake_image = decoder(r, all_inds)

                    disc_real = critic(tiled_ground_truth) # we made need to change this to tiles?
                    disc_fake = critic(fake_image)
                    gradient_penalty = utils.calc_gradient_penalty(critic, tiled_ground_truth, fake_image)
                    loss = disc_fake - disc_real + gradient_penalty
                    loss.backward()
                    w_dist = disc_real - disc_fake
                optimizer_critic.step()
            progress.set_description("E{} - L{:.4f}".format(epoch, w_dist.item()))

            with open("encoder.pkl", "wb") as of:
                pickle.dump(encoder, of)

            with open("decoder.pkl", "wb") as of:
                pickle.dump(decoder, of)
            break

        encoder.eval()
        decoder.eval()
        with torch.no_grad():

            for i, (ground_truth_image, target) in enumerate(test_loader):
                ground_truth_image = ground_truth_image.view(3, 32, 32)

                tiles, inds, frame, all_inds, tiled_ground_truth = utils.get_tiles(ground_truth_image, num_tiles=32)

                tiles = tiles.float().to(device)
                inds = inds.float().to(device)
                all_inds = all_inds.float().to(device)
                tiled_ground_truth = tiled_ground_truth.float().to(device)  

                r = encoder(inds, tiles)
                generated_image = decoder(r, all_inds)

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
                        ax[row][col].imshow(tiled_ground_truth[count].detach().cpu())
                        count += 1
                plt.axis("off")
                plt.savefig("{}groundtruth{}.png".format(epoch,i))
                plt.close()

                if i >= 10:
                    break
