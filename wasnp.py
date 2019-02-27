import matplotlib.pyplot as plt
import numpy as np

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import networkx as nx

from tqdm import tqdm

import os

import utils

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
    

class Encoder(nn.Module):
    """takes in context points and returns a fixed length aggregation"""
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 128)
        

    def forward(self, x):
        """x = sparsely sampled image
        this returns the aggregated r value
        """
        cntx = data.nonzero()
        x_points = cntx[:,0]
        y_points = cntx[:,1]
        
        intensities = x[x_points, y_points]
        
        output = torch.empty((intensities.shape[0], 128)).to(device)

        intensities = torch.stack((x_points.float(), y_points.float(), intensities.float()))
        intensities.transpose_(0,1)


        for i, val in enumerate(intensities):
            output[i] = self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(val)))))))
            
        return output.mean(0).view(1, 128)

class Decoder(nn.Module):
    def __init__(self, m, n):
        super(Decoder, self).__init__()
        self.m = m
        self.n = n
        self.fc1 = nn.Linear(130, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)
        
    def forward(self, r):
        """r is the aggregated data used to condition"""
        # we only take in r, because in this case x is all points in size of image (28,28)
        x = torch.tensor([[i, j] for i in range(0,self.m) for j in range(0,self.n)]).float().to(device)
        out = torch.cat((x, r.view(1,-1).repeat(1,self.m*self.n).view(self.m*self.n,128)), 1)

        h = self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(out)))))))

        mu = h[:,0]
        log_sigma = h[:,1]
        
        # bound the variance
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)
        
        return mu, sigma

if __name__ == "__main__":

    batch_size=1
    test_batch_size=1000

    m, n = 28, 28
    num_pixels = m*n

    min_context_points = num_pixels * 0.15 # always have at least 15% of all pixels
    max_context_points = num_pixels * 0.95 # always have at most 95% of all pixels

    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    kwargs = {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)),
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)


    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)),
                       ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    epochs = 10
    log_interval = 50

    encoder = Encoder().to(device)
    decoder = Decoder(m, n).to(device)

    encoder = nn.DataParallel(encoder)
    decoder = nn.DataParallel(decoder)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))

    for epoch in range(1, epochs+1):
        encoder.train()
        decoder.train()

        progress = tqdm(enumerate(train_loader))

        total_loss = 0
        count = 0
        for i, (ground_truth_image, target) in progress: # we don't use target, because this is more unsupervised
            ground_truth_image = ground_truth_image.view(28, 28)

            data = utils.get_mnist_context_points(ground_truth_image, context_points=np.random.randint(min_context_points, max_context_points))

            optimizer.zero_grad()

            data = data.to(device).float()
            
            # run the model to get r which will be concatenated onto every node pair in the decoder
            r = encoder(data)

            mu, sigma = decoder(r)

            mu = mu.view(m,n)
            sigma = sigma.view(m,n)

            loss = -utils.get_log_p(data, mu, sigma).mean()

            total_loss += loss
            count += 1

            loss.backward()
            optimizer.step()
            progress.set_description('E:{} - Loss: {:.4f}'.format(epoch, total_loss/count))

            if i >= 1000:
                break
        
        # with open("encoder_mnist.pkl", "wb") as of:
        #     pickle.dump(encoder, of)

        # with open("encoder_mnist.pkl", "wb") as of:
        #     pickle.dump(decoder, of)

        # with open("optim.pkl", "wb") as of:
        #     pickle.dump(optimizer, of)

        encoder.eval()
        decoder.eval()
        with torch.no_grad():

            for i, (ground_truth_image, target) in enumerate(train_loader):
                ground_truth_image = ground_truth_image.view(28, 28)

                data = utils.get_mnist_context_points(ground_truth_image, context_points=400)

                
                data = data.to(device)

                r = encoder(data)

                mu, sigma = decoder(r)


                plt.imshow(data.reshape(m,n), cmap='gray')
                plt.axis("off")
                plt.title("context points")
                plt.savefig("{}context_points{}.png".format(epoch, i), dpi=300)
                plt.close()

                plt.imshow(ground_truth_image.reshape(m,n), cmap='gray')
                plt.axis("off")
                plt.title("ground truth")
                plt.savefig("{}ground_truth{}.png".format(epoch, i), dpi=300)
                plt.close()

                plt.imshow(mu.detach().reshape(m,n), cmap='gray')
                plt.axis("off")
                plt.title("mean")
                plt.savefig("{}mean{}.png".format(epoch, i), dpi=300)
                plt.close()

                plt.imshow(sigma.detach().reshape(m, n), cmap='gray')
                plt.axis("off")
                plt.title("variance")
                plt.savefig("{}var{}.png".format(epoch, i), dpi=300)
                plt.close()

                if i >= 10:
                    break


