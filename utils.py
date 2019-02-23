import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import torch

import scipy

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import supervised

def load__dataset_from_pickle(path="./mutag.pkl"):
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

def draw_graph(G, title="", save=False):

    pos = nx.spring_layout(G)

    nx.draw(G, pos)
    node_labels = nx.get_node_attributes(G,'node_value')
    nx.draw_networkx_labels(G, pos, labels = node_labels)


    edge_labels = nx.get_edge_attributes(G,'edge_value')
    if type(list(edge_labels.items())[0][1]) == torch.Tensor and list(edge_labels.items())[0][1].shape[0] == 1:
        edge_labels = {k:round(v.item(), 4) for i, (k, v) in enumerate(edge_labels.items())}
    elif type(list(edge_labels.items())[0][1]) == torch.Tensor and list(edge_labels.items())[0][1].shape[0] == 4:
        edge_labels = {k:round(torch.argmax(v, dim=0).item(), 4) for i, (k, v) in enumerate(edge_labels.items())}

    
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    if save:
        plt.savefig("{}".format(title))
        plt.close()
    else:
        plt.show()

def get_data(path='./mutag.pkl', train_size=.7):
    graphs = convert_to_network(load__dataset_from_pickle(path))
    train, test = train_test_split(graphs, train_size=train_size)
    return train, test # lists of networkx graphs

def get_edge_matrix(G, return_tensor=False):
    if return_tensor:
        edge_value = lambda u, v: G[u][v]['edge_value']
        node_value = lambda u: u

        ordering = list(set([node_value(n) for n in G]))

        N = len(ordering)
        undirected = not G.is_directed()   
        index = dict(zip(ordering, range(N)))
        M = torch.zeros((N,N))
        M_ = torch.zeros((N,N,4))
        use_extended = False
        seen = set([])

        for u,nbrdict in G.adjacency():
            for v in nbrdict:
                # Obtain the node attribute values.
                i, j = index[node_value(u)], index[node_value(v)]
                if v not in seen:
                    if type(edge_value(u,v)) == torch.Tensor and edge_value(u,v).shape[0] == 1:
                        M[i,j] += edge_value(u,v).item()
                    elif type(edge_value(u,v)) == torch.Tensor and edge_value(u,v).shape[0] == 4:
                        M_[i,j] = edge_value(u,v)
                        use_extended = True
                    else:
                        M[i,j] += edge_value(u,v)

                    if undirected:
                        M[j,i] = M[i,j]

            if undirected:
                seen.add(u)    

        if use_extended:
            return M_
        else:
            return M
    else:
        return nx.attr_matrix(G, edge_attr='edge_value')[0]

def sparsely_observe_graph(G, min_context_percent, max_context_percent):
    out_G = G.copy()
    edges = list(out_G.edges)
    n = len(edges)

    mask = np.zeros(n)
    
    context_points = np.random.randint(int(n*min_context_percent), int(n*max_context_percent))

    mask[:context_points] = 1
    np.random.shuffle(mask)
    inds = np.where(mask == 1)[0]

    for i in inds:
        out_G.remove_edge(edges[i][0], edges[i][1])

    return out_G

def graph_neighbor_feature_extractor(G, target=False):
    """
    parameters - 
    G (networkx.classes.graph.Graph) - graph with proper edge_value, node_value structure as defined in convert_to_network
    
    outputs - 
    T (torch.Tensor) - tensor of shape #unknown n_edges x [node1_val, node2_val, node1_degree, node2_degree]
    """

    data = []
    target_values = []
    for edge in G.edges:
        neighborhood_values = []
        node1, node2 = edge

        for edge_ in list(G.edges(node1)):
            try:
                neighborhood_values.append(nx.get_edge_attributes(G, 'edge_value')[edge_])
            except:
                neighborhood_values.append(nx.get_edge_attributes(G, 'edge_value')[edge_[::-1]])

        for edge_ in list(G.edges(node2)):
            try:
                neighborhood_values.append(nx.get_edge_attributes(G, 'edge_value')[edge_])
            except:
                neighborhood_values.append(nx.get_edge_attributes(G, 'edge_value')[edge_[::-1]])

        if target:
            target_values.append(G.edges[edge]['edge_value'])

        data.append(neighborhood_values)

    if target:
        return data, target_values
    else:
        return data

def graph_to_tensor_feature_extractor(G, target=False):
    """
    parameters - 
    G (networkx.classes.graph.Graph) - graph with proper edge_value, node_value structure as defined in convert_to_network
    
    outputs - 
    T (torch.Tensor) - tensor of shape #unknown n_edges x [node1_val, node2_val, node1_degree, node2_degree]
    """
    A = nx.adjacency_matrix(G).toarray()

    N = A.shape[0]
    diags = A.sum(axis=1)**(1/2)
    D = scipy.sparse.spdiags(diags.flatten(), [0], N, N, format='csr').toarray()
    
    L = np.eye(N) - D.dot(A).dot(D) # calculate normalized graph laplacian
    val, vec = np.linalg.eig(L)
    g_nodes = list(G.nodes())

    data = []
    target_values = []
    for edge in G.edges:
        node1, node2 = edge
        ind1, ind2 = g_nodes.index(node1), g_nodes.index(node2)
        node1_val, node2_val = G.nodes[node1]['node_value'], G.nodes[node2]['node_value']
        node1_degree, node2_degree = G.degree[node1], G.degree[node2]
        eigs1, eigs2 = vec[ind1][:10], vec[ind2][:10]
        features = [node1_val, node2_val, node1_degree, node2_degree]
        features.extend(eigs1)
        features.extend(eigs2)

        if target:
            target_values.append(G.edges[edge]['edge_value'])

        data.append(features)

    if target:
        return torch.Tensor(data), torch.Tensor(target_values)
    else:
        return torch.Tensor(data)

def graph_to_tensor_four_features(G):
    """
    depreciated
    parameters - 
    G (networkx.classes.graph.Graph) - graph with proper edge_value, node_value structure as defined in convert_to_network
    
    outputs - 
    T (torch.Tensor) - tensor of shape n_edges x [node1_val, node2_val, node1_degree, node2_degree]
    """
    data = []
    for edge in G.edges:
        node1, node2 = edge
        node1_val, node2_val = G.nodes[node1]['node_value'], G.nodes[node2]['node_value']
        node1_degree, node2_degree = G.degree[node1], G.degree[node2]
        data.append([node1_val, node2_val, node1_degree, node2_degree])

    return torch.Tensor(data)


def reconstruct_graph(edges, graph, return_tensor=False):
    """
    parameters - 
    edges (torch.Tensor) - edges that you have predicted for the target node values
    graph (nx.classes.graph.Graph) - full target graph, we will only be using the node values
    return_tensor (bool) - choose what type to return


    outputs -
    approximate_graph (nx.classes.graph.Graph or torch.Tensor) - graph with node values from target and edges from edges, node target has edge values already, we overwrite those
    """
    approximate_graph = graph.copy()


    # get the dictionary of the proper shape
    edge_data = nx.get_edge_attributes(approximate_graph, 'edge_value')

    # replace the values with the inputed values from out edges list, the order should be fine, but we could check if it doesn't work
    edge_data = {k:edges[i] for i, (k, v) in enumerate(edge_data.items())}

    nx.set_edge_attributes(approximate_graph, edge_data, 'edge_value')


    if return_tensor:
        return graph_to_tensor_four_features(approximate_graph)
    else:
        return approximate_graph

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def distance_metric(G1, G2):
    """get the distance between two graphs, using the value of the edges, assuming node values are the same"""
    g1_edges = np.array(torch.nn.functional.softmax(get_edge_matrix(G1, return_tensor=True).detach(), dim=1))
    g2_edges = np.array(get_edge_matrix(G2, return_tensor=True).detach())
    if g1_edges.shape == g2_edges.shape:
        return np.linalg.norm(g1_edges - g2_edges, ord=2)
    
    return np.linalg.norm(np.argmax(g1_edges, -1) - g2_edges, ord=2) # sqrt ( sumi ( sumj (aij - bij) **2)), we can change this if needs be

def get_accuracy(y_hat, y, as_dict=False, acc=False):
    # y_hat_edges = {k:np.argmax(softmax(np.array(v.detach()))) for k,v in nx.get_edge_attributes(y_hat, 'edge_value').items()}
    # y_edges = nx.get_edge_attributes(y, 'edge_value')
    predicted = np.argmax(softmax(np.array(y_hat.detach())), axis=1)
    target = np.array(y.detach())

    if acc:
        return classification_report(target, predicted, output_dict=as_dict), accuracy_score(target, predicted)
    else:
        return classification_report(target, predicted, output_dict=as_dict)

def run_baselines(train, test, outfile_name="full_baseline"):
    datum = {"rf_cr": [], "rf_acc": [], "random_cr": [], "random_acc": [], "common_cr": [], "common_acc": []}

    rf_cr, rf_acc = supervised.train_rf(train, test)
    random_cr, random_acc = supervised.train_random(train, test)
    common_cr, common_acc = supervised.train_common(train, test)
    datum['rf_cr'].append(rf_cr)
    datum['rf_acc'].append(rf_acc)
    datum['random_cr'].append(random_cr)
    datum['random_acc'].append(random_acc)
    datum['common_cr'].append(common_cr)
    datum['common_acc'].append(common_acc)

    with open("{}.pkl".format(outfile_name), "wb") as f:
        pickle.dump(datum, f)

if __name__ == "__main__":
    d = load_dataset_from_pickle()
    graphs = convert_to_network(d)

    G = graphs[90]
    draw_graph(G)
    