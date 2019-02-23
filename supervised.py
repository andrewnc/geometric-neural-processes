from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

import numpy as np
import networkx as nx

import utils

from collections import Counter
import pickle
from tqdm import tqdm


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

def train_common_neighbors(train, test):
   

    test_combined = [utils.graph_neighbor_feature_extractor(x, target=True) for x in test]

    test_x, test_y = [],[]
    y_pred = []

    for (data, target) in test_combined:
        test_x.extend(np.array(data))
        test_y.extend(np.array(target))

    # mask = np.zeros(len(test_x))
    # mask[: np.random.randint(int(len(test_x)*0.75), int(len(test_x)*0.9))] = 1
    # np.random.shuffle(mask)

    for i in range(len(test_x)):
        # if mask[i] == 0:
        #     y_pred.append(test_y[i])
        # else:
        y_pred.append(Counter(test_x[i]).most_common(1)[0][0])

    return classification_report(test_y, y_pred, output_dict=True), accuracy_score(test_y, y_pred)




if __name__ == "__main__":
    datum = {"rf_cr": [], "rf_acc": [], "random_cr": [], "random_acc": [], "common_cr": [], "common_acc": [], "neigh_cr": [], "neigh_acc": [], "data_amount": []}
    for i in tqdm(range(50)):
        train_true, test = utils.get_data(train_size=.7)
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
    