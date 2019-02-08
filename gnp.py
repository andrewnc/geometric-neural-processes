import matplotlib.pyplot as plt
import math

import numpy as np

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

import os

import time

import utils


m,n = 224,224 #28, 28
batch_size = 1

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
    

def get_log_p(data, mu, sigma):
    return -torch.log(torch.sqrt(2*math.pi*(sigma**2))) - (data - mu)**2/(2*(sigma**2))

def normal_kl(mu1, sigma1, mu2, sigma2):
    return 1/2 * ((1 + torch.log(sigma1**2) - mu1**2 - sigma1**2)+(1 + torch.log(sigma2**2) - mu2**2 - sigma2**2))



class GraphEncoder(nn.Module):
    """we somehow need to use the idea of a graph convolution to gather local features"""
    def __init__(self):
        super(GraphEncoder, self).__init__()
        pass
    def forward(self, x):
        return x

class NonGraphEncoder(nn.Module):
    """In this encoder we are assuming that you sparesly observe the edge values and we are trying 
    to infill these values. Also, are using the most naive value that you could imagine"""
    def __init__(self):
        super(NonGraphEncoder, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 128)
        

    def forward(self, x):
        """x = num_context x [node1_val, node2_val, node1_degree, node2_degree]
        this returns the aggregated r value
        """
        return torch.mean(self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))), dim=0) # aggregate




class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(132, 64) # 128 vector with 4 features, see NonGraphEncoder.forward doc strings
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, r, G):
        """r is the aggregated data used to condition"""
        # we need to take in G in this case because we need to feed in the node values to the decoder
        # we must be very careful not to give data to the decoder that is supposed to be masked
        x = []
        for edge in G.edges:
            node1, node2 = edge
            node1_val, node2_val = G.nodes[node1]['node_value'], G.nodes[node2]['node_value']
            node1_degree, node2_degree = G.degree[node1], G.degree[node2]
            x.append([node1_val, node2_val, node1_degree, node2_degree])

        x = torch.tensor(x)
        
        # I might not need the "view" portion of this first
        x = torch.cat((x, r.view(1,-1).repeat(1,len(G.edges)).view(len(G.edges),128)), 1)
        
        h = self.fc4(F.relu(self.fc3(F.relu(self.fc2_5(F.relu(self.fc2(F.relu(self.fc1(x)))))))))
        
        return h # this should be [n_context x 1]


if __name__ == "__main__":
    min_context_percent = 0.1
    max_context_percent = 0.9

    epochs = 10

    log_interval = 50

    encoder = NonGraphEncoder().to(device)
    decoder = Decoder().to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))
    train, test = utils.get_data(path='./mutag.pkl')

    for epoch in range(1, epochs+1):
        encoder.train()
        decoder.train()

        progress = tqdm(enumerate(train))

        for i, graph in progress:

            sparse_graph = utils.sparsely_observe_graph(graph, min_context_percent, max_context_percent)

            

            target = graph

            optimizer.zero_grad()

            data = data.to(device)
            target = target.to(device)
            
            # run the model to get r
            r = encoder(data)
            edges = decoder(r)

            print(edges)


            ## we need to reconstruct the graph
            log_p = get_log_p(target, mu, sigma)

            loss = -log_p.mean()
            loss.backward()
            optimizer.step()

            if i % log_interval == 0:
                progress.set_description('{} - Loss: {:.4f} Mean: {:.3f}/{:.3f} Sig: {:.3f}/{:.3f}'.format(epoch, loss.item(), mu.max(), mu.min(), sigma.max(), sigma.min()))
                with open("encoder_graph.pkl", "wb") as of:
                    pickle.dump(encoder, of)

                with open("encoder_graph.pkl", "wb") as of:
                    pickle.dump(decoder, of)

                with open("optim.pkl", "wb") as of:
                    pickle.dump(optimizer, of)

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for i, data in enumerate(test_x):

                target = test_y[i]

                data = data.to(device)
                target = target.to(device)

                r = encoder(data)

                mu, sigma = decoder(r)

                try:
                    mu = mu.view(batch_size, m, n)
                    sigma = sigma.view(batch_size, m,n)
                except Exception as e:
                    print("could not resize")
                    print(e)
                    continue


                try:
                    plt.imsave("{}target{}.png".format(epoch, i), target[0].detach().view(m,n))

                except Exception as e:
                    print("could not save first target")
                    print(e)

                try:
                    data = data.transpose(1,2).transpose(2, 3)
                    plt.imsave("{}masked_data{}.png".format(epoch, i), slice_and_dice(data[0][:,:,0]))
                except Exception as e:
                    print("could not transpose, or slice and dice first")
                    print(e)

                try:
                    plt.imsave("{}mean{}.png".format(epoch, i), mu[0].detach())
                    plt.imsave("{}var{}.png".format(epoch, i), sigma[0].detach())
                except Exception as e:
                    print("could not save mean or var first")
                    print(e)