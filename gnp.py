import matplotlib.pyplot as plt
import math

import numpy as np

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import networkx as nx

from tqdm import tqdm

import os

import time

import utils
from layers import GraphConvolution
from supervised import train_rf

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
        self.gc1 = GraphConvolution(17, 17)
        self.gc2 = GraphConvolution(17, 17)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        print(x.shape)
        return x

class NonGraphEncoder(nn.Module):
    """In this encoder we are assuming that you sparesly observe the edge values and we are trying 
    to infill these values. Also, are using the most naive value that you could imagine"""
    def __init__(self):
        super(NonGraphEncoder, self).__init__()
        self.fc1 = nn.Linear(24, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        

    def forward(self, x):
        """x = num_context x [node1_val, node2_val, node1_degree, node2_degree]
        this returns the aggregated r value
        """
        return torch.mean(self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))), dim=0) # aggregate


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(280, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 4)
        
    def forward(self, r, x):
        """r is the aggregated data used to condition"""
        # we need to take in G (as x value) in this case because we need to feed in the node values to the decoder
        # we must be very careful not to give data to the decoder that is supposed to be masked
        
        n = x.shape[0] # number of edges

        out = torch.cat((x, r.view(1,-1).repeat(1,n).view(n,256)), 1)
        
        h = self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(out)))))))
        
        return h # this should be [n_edges x 4]


if __name__ == "__main__":

    results = []

    input_data_paths = os.listdir("./input_graph_datasets") # this is the list of all datasets we have
    path = "tox21_ahr.pkl"
    
    #filter graphs min keyword will remove graphs with fewer nodes than the value passed in. Set this value equal to m (the slice size in utils.graph_to_tensor_feature_extractor)
    train, test = utils.get_data(path="./input_graph_datasets/" +path, filter_graphs_min=10) 

    min_context_percent = 0.4
    max_context_percent = 0.9

    epochs = 10

    log_interval = 50

    encoder = NonGraphEncoder().to(device)
    # encoder = GraphEncoder().to(device)
    decoder = Decoder().to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))
    # loss = nn.CrossEntropyLoss(weight=torch.tensor([1/2000, 1/1000, 1/300, 1/10]).float().to(device))
    loss = nn.CrossEntropyLoss()

    # subsampled_train = train[:data_amount] # subsample to compare
    
    for eigen_feature in range(10):
        all_metrics = []
        for graph_m in range(1, 132):
            for epoch in range(1, epochs+1):
                encoder.train()
                decoder.train()

                np.random.shuffle(train) # change the order of the training data

                progress = tqdm(enumerate(train))

                total_loss = 0
                total_p, total_r, total_f1, total_acc = 0,0,0,0
                count = 0
                for i, graph in progress:
                    full_loss =0 

                    for j in range(50):

                        sparse_graph = utils.sparsely_observe_graph(graph, min_context_percent, max_context_percent)

                        # this acts as the feature extractor from graph to data...
                        data = utils.graph_to_tensor_feature_extractor(sparse_graph, m=graph_m)
                        target, graph_edge = utils.graph_to_tensor_feature_extractor(graph,m=graph_m, target=True)

                        optimizer.zero_grad()

                        data = data.to(device)
                        target = target.to(device)
                        graph_edge = graph_edge.to(device)
                        
                        # run the model to get r which will be concatenated onto every node pair in the decoder
                        r = encoder(data)

                        edges = decoder(r, target) # yes, it takes in the target, but doesn't use any edge values from the target

                        approximate_graph = utils.reconstruct_graph(edges, graph)

                        loss_val = loss(edges.float(), graph_edge.long())

                        total_loss += loss_val.item()
                        count += 1
                        acc, out_acc = utils.get_accuracy(edges, graph_edge, as_dict=True, acc=True)
                        total_p += acc['weighted avg']['precision'] 
                        total_r += acc['weighted avg']['recall']
                        total_f1 += acc['weighted avg']['f1-score']
                        total_acc += out_acc
                        loss_val.backward()
                        optimizer.step()
                    
                    with open("encoder_graph.pkl", "wb") as of:
                        progress.set_description('E:{} - Loss: {:.4f} P: {:.4f} R: {:.4f} F1: {:.4f} A: {:.4f}'.format(epoch, total_loss/count, total_p/count, total_r/count, total_f1/count, total_acc/count))
                        pickle.dump(encoder, of)

                    with open("encoder_graph.pkl", "wb") as of:
                        pickle.dump(decoder, of)

                    with open("optim.pkl", "wb") as of:
                        pickle.dump(optimizer, of)

            encoder.eval()
            decoder.eval()
            with torch.no_grad():

                metrics = {"precision": [],"recall": [],"f1-score":[], "accuracy": []}
                for i, graph in enumerate(test):

                    
                    sparse_graph = utils.sparsely_observe_graph(graph, .75, .9)
                    data = utils.graph_to_tensor_feature_extractor(sparse_graph)

                    target, graph_edge = utils.graph_to_tensor_feature_extractor(graph, target=True)

                    data = data.to(device)
                    target = target.to(device)

                    r = encoder(data)

                    edges = decoder(r, target)

                    # approximate_graph = utils.reconstruct_graph(edges, graph)
                    classification_report, accuracy = utils.get_accuracy(edges, graph_edge, as_dict = True, acc=True)

                    metrics['precision'].append(classification_report['weighted avg']['precision']) 
                    metrics['recall'].append(classification_report['weighted avg']['recall']) 
                    metrics['f1-score'].append(classification_report['weighted avg']['f1-score'])
                    metrics['accuracy'].append(accuracy)


                    # utils.draw_graph(graph, title="{}target{}.png".format(data_amount, i), save=True)
                    # utils.draw_graph(approximate_graph, title="{}reconstruction{}.png".format(data_amount, i), save=True)
                print("precision {}".format(np.mean(metrics["precision"])))
                print("recall {}".format(np.mean(metrics["recall"])))
                print("f1-score {}".format(np.mean(metrics["f1-score"])))
                print("accuracy {}".format(np.mean(metrics["accuracy"])))


                # ---------------- uncomment this one line below to run the baselines ------------------------ #
                # utils.run_baselines(train, test, outfile_name="{}full_baseline".format(path))
            
            results.append({"gnp_cr":metrics, "gnp_acc":accuracy})    
            with open("{}results.pkl".format(path), "wb") as f:
                pickle.dump(results, f)
        all_metrics.append([graph_m, results])
        with open("all_metrics.pkl", "wb") as f:
            pickle.dump(all_metrics, f)

