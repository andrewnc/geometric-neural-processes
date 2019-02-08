import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle

from sklearn.model_selection import train_test_split


def load_mutag_dataset_from_pickle(path="./mutag.pkl"):
    """path (str): full path to MUTAG pickle file. EG /home/user/Downloads/mutag.pkl"""
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d

def convert_to_network(d):
    """d (sklearn.utils.Bunch) dataset loaded from pkl with keys data, target"""
    graphs = []
    for item in d.data:
        """item[0] is the connections between nodes
           item[1] is the value at each node
           item[2] is the weight on each edge
        """
        G = nx.Graph(list(item[0]))
        nx.set_node_attributes(G, item[1], "node_value")
        nx.set_edge_attributes(G, item[2], "edge_value")
        graphs.append(G)
    return graphs

def draw_graph(G):
    pos = nx.spring_layout(G)

    nx.draw(G, pos)
    node_labels = nx.get_node_attributes(G,'node_value')
    nx.draw_networkx_labels(G, pos, labels = node_labels)
    edge_labels = nx.get_edge_attributes(G,'edge_value')
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    plt.show()

def get_data(path='./mutag.pkl'):
    graphs = convert_to_network(load_mutag_dataset_from_pickle())
    train, test = train_test_split(graphs)
    return train, test # lists of networkx graphs

def get_edge_matrix(G):
    return np.array(nx.attr_matrix(G, edge_attr='edge_value')[0])

def sparsely_observe_graph(G, min_context_percent, max_context_percent):
    edges = list(G.edges)
    n = len(edges)

    mask = np.zeros(n)
    
    context_points = np.random.randint(int(n*min_context_percent), int(n*max_context_percent))

    mask[:context_points] = 1
    np.random.shuffle(mask)
    inds = np.where(mask == 1)[0]

    for i in inds:
        G.remove_edge(edges[i][0], edges[i][1])

    return G

def graph_to_tensor

def distance_metric(G1, G2):
    """get the distance between two graphs, using the value of the edges, assuming node values are the same"""
    g1_edges = get_edge_matrix(G1)
    g2_edges = get_edge_matrix(G2)
    return np.linalg.norm(g1_edges - g2_edges, ord=2) # sqrt ( sumi ( sumj (aij - bij) **2)), we can change this if needs be



if __name__ == "__main__":
    d = load_mutag_dataset_from_pickle()
    graphs = convert_to_network(d)

    G = graphs[90]
    draw_graph(G)
    