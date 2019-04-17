from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

import numpy as np
import networkx as nx

import utils

from collections import Counter
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os

class Net(nn.Module):
    """In this encoder we are assuming that you sparesly observe the edge values and we are trying 
    to infill these values. Also, are using the most naive value that you could imagine"""
    def __init__(self, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(24, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 32)
        self.fc8 = nn.Linear(32, output_size)
        


    def forward(self, x):
        """x = num_context x [node1_val, node2_val, node1_degree, node2_degree]
        this returns the aggregated r value
        """
        encoder_portion = F.relu(self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))))
        out = self.fc8(F.relu(self.fc7(F.relu(self.fc6(F.relu(self.fc5(encoder_portion)))))))

        return out


def train_nn(train, test):
    train_combined = [utils.graph_to_tensor_feature_extractor(utils.sparsely_observe_graph(x, .4, .9), target=True) for x in train]
    test_combined = [utils.graph_to_tensor_feature_extractor(x, target=True) for x in test]


    train_x, train_y = [],[]
    test_x, test_y = [],[]

    for (data, target) in train_combined:
        train_x.extend(np.array(data))
        train_y.extend(np.array(target))

    for (data, target) in test_combined:
        test_x.extend(np.array(data))
        test_y.extend(np.array(target))
    n = int(max(train_y) + 1)
    one_hot_target_y = utils.indices_to_one_hot(train_y, n)
    network = Net(n).to("cuda")
    optimizer = optim.Adam(network.parameters())
    loss = nn.CrossEntropyLoss()

    network.train()
    for epoch in range(2):
        optimizer.zero_grad()
        total_loss = 0
        progress = tqdm(enumerate(train_combined))
        for i, (data, target) in progress:
            data = data.to("cuda")
            output = network(data)
            loss_val = loss(output.float(), target.to("cuda").long())
            loss_val.backward()
            optimizer.step()
            total_loss += loss_val.item()
            progress.set_description("epoch {} - loss {}".format(epoch, total_loss/(i+1)))

    network.eval()
    pred_y = []
    with torch.no_grad():
        for (data, target) in test_combined:
            data = data.to("cuda")
            output = network(data)
            pred_y.extend(np.array(np.argmax(F.softmax(output, dim=1).detach().cpu(), axis=1)))
    return classification_report(test_y, pred_y, output_dict=True), accuracy_score(test_y, pred_y)


def train_random(train, test):
    test_combined = [utils.graph_to_tensor_feature_extractor(x, target=True) for x in test]

    test_x, test_y = [],[]

    for (data, target) in test_combined:
        test_x.extend(np.array(data))
        test_y.extend(np.array(target))


    y_pred = np.random.randint(0, 4, size=len(test_y))

    return classification_report(test_y, y_pred, output_dict=True), accuracy_score(test_y, y_pred)

def train_common(train, test):
    train_combined = [utils.sparsely_observe_graph(x, .4, .9) for x in train]
    
    test_sparse = [utils.graph_to_tensor_feature_extractor(utils.sparsely_observe_graph(x, .75, .9), target=True) for x in test]

    test_combined = [utils.graph_to_tensor_feature_extractor(x, target=True) for x in test]

    test_x, test_y = [],[]

    for (data, target) in test_combined:
        test_x.extend(np.array(data))
        test_y.extend(np.array(target))

    common_value = Counter([Counter(list(nx.get_edge_attributes(x, 'edge_value').values())).most_common(1)[0][0] for x in train_combined]).most_common(1)[0][0]

    y_pred = [common_value]*len(test_y)

    return classification_report(test_y, y_pred, output_dict=True), accuracy_score(test_y, y_pred)

def train_rf(train, test):
    train_combined = [utils.graph_to_tensor_feature_extractor(utils.sparsely_observe_graph(x, .4, .9), target=True) for x in train]
    test_combined = [utils.graph_to_tensor_feature_extractor(x, target=True) for x in test]


    train_x, train_y = [],[]
    test_x, test_y = [],[]

    for (data, target) in train_combined:
        train_x.extend(np.array(data))
        train_y.extend(np.array(target))

    for (data, target) in test_combined:
        test_x.extend(np.array(data))
        test_y.extend(np.array(target))

    clf = RandomForestClassifier()


    clf.fit(train_x, train_y)


    y_pred = clf.predict(test_x)

    return classification_report(test_y, y_pred, output_dict=True), accuracy_score(test_y, y_pred)


def train_common_neighbors(train, test):
    for_common = [utils.sparsely_observe_graph(x, .4, .9) for x in train]
    train_combined = [utils.graph_neighbor_feature_extractor(utils.sparsely_observe_graph(x, .4, .9), target=True) for x in test]
    test_combined = [utils.graph_neighbor_feature_extractor(x, target=True) for x in test]
    test_x, test_y = [],[]
    train_x, train_y = [],[]
    y_pred = []

    for (data, target) in train_combined:
        train_x.append(data)
        train_y.extend(np.array(target))

    for (data, target) in test_combined:
        test_x.append(data)
        test_y.extend(np.array(target))

    max_val = max(test_y) + 1

    common_value = Counter([Counter(list(nx.get_edge_attributes(x, 'edge_value').values())).most_common(1)[0][0] for x in for_common]).most_common(1)[0][0]
    for i, dict_ in enumerate(test_x):
        for key, value in dict_.items():
            try:
                y_pred.append(Counter(train_x[i][key]).most_common(1)[0][0])
            except Exception as e:
                y_pred.append(np.random.randint(0, max_val))
    return classification_report(test_y, y_pred, output_dict=True), accuracy_score(test_y, y_pred)


def run_all(train, test):
    for data_amount in [10, 70, 130, 169]:
        train = train_true[:data_amount]
        rf_cr, rf_acc = train_rf(train, test)
        random_cr, random_acc = train_random(train, test)
        common_cr, common_acc = train_common(train, test)
        neigh_cr, neigh_acc = train_common_neighbors(train, test)
        datum['rf_cr'].append(rf_cr)
        datum['rf_acc'].append(rf_acc)
        datum['random_cr'].append(random_cr)
        datum['random_acc'].append(random_acc)
        datum['common_cr'].append(common_cr)
        datum['common_acc'].append(common_acc)
        datum['neigh_cr'].append(neigh_cr)
        datum['neigh_acc'].append(neigh_acc)
        datum['data_amount'].append(data_amount)

    with open("full_baseline.pkl", "wb") as f:
        pickle.dump(datum, f)
    
if __name__ == "__main__":
    dir_ = "./input_graph_datasets/"
    full_data = {f:[] for f in os.listdir(dir_)}
    for f in tqdm(os.listdir(dir_)):
        if f.split(".")[-1] != "pkl":
            continue
        datum = {"neighbor_cr": [], "neighbor_acc":[], "nn_cr": [], "nn_acc": [], "rf_cr": [], "rf_acc": [], "random_cr": [], "random_acc": [], "common_cr": [], "common_acc": [], "neigh_cr": [], "neigh_acc": [], "data_amount": []}
        for i in tqdm(range(10)):
            train_true, test = utils.get_data(dir_ + f, train_size=.7, filter_graphs_min=10)
            nn_cr, nn_acc = train_common_neighbors(train_true, test)
            datum["neighbor_cr"].append(nn_cr)
            datum["neighbor_acc"].append(nn_acc)
            print(nn_acc)
        full_data[f].append(datum)
    with open("neighbor_common_baseline.pkl", "wb") as wf:
        pickle.dump(full_data, wf)


        