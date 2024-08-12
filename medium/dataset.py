import os
import pickle as pkl
from os import path

import networkx as nx
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from data_utils import normalize_feat, rand_train_test_idx, split_data
from sklearn.preprocessing import label_binarize
from torch_geometric.datasets import Planetoid

DATAPATH = '../../data/'


class NCDataset(object):
    def __init__(self, name, root=f'{DATAPATH}'):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}

        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_nc_dataset(args):
    """ Loader for NCDataset
        Returns NCDataset
    """
    global DATAPATH

    DATAPATH = args.data_dir
    dataname = args.dataset
    print('>> Loading dataset: {}'.format(dataname))
    if dataname == 'deezer-europe':
        dataset = load_deezer_dataset()

    elif dataname in ('cora', 'citeseer', 'pubmed'):
        dataset = load_planetoid_dataset(dataname, args.no_feat_norm)

    elif dataname in ('film'):
        dataset = load_geom_gcn_dataset(dataname)

    elif dataname in ('chameleon', 'squirrel'):
        dataset = load_wiki_new(dataname, args.no_feat_norm)
        # dataset = load_wikipedia(dataname,args.no_feat_norm)
    elif dataname == 'airport':
        dataset = load_airport_dataset()

    elif dataname == 'disease':
        dataset = load_disease_dataset()

    elif dataname == '20news':
        dataset = load_20news()

    elif dataname == 'mini':
        dataset = load_mini_imagenet()

    else:
        raise ValueError('Invalid dataname')
    return dataset


def load_deezer_dataset():
    filename = 'deezer-europe'
    dataset = NCDataset(filename)
    deezer = scipy.io.loadmat(f'{DATAPATH}/deezer/deezer-europe.mat')

    A, label, features = deezer['A'], deezer['label'], deezer['features']
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset


def load_airport_dataset():
    filename = 'airport'
    dataset = NCDataset(filename)
    graph = pkl.load(open(os.path.join(DATAPATH, 'hgcn_data', 'airport', 'airport.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph._node[u]['feat'] for u in graph.nodes()])
    label_idx = 4
    labels = features[:, label_idx]
    features = features[:, :label_idx]
    labels = bin_feat(labels, bins=[7.0 / 7, 8.0 / 7, 9.0 / 7])
    num_nodes = adj.shape[0]
    features = torch.tensor(features, dtype=torch.float)
    val_prop, test_prop = 0.15, 0.15
    idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop)
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': features,
                     'num_nodes': num_nodes}
    dataset.label = labels
    dataset.train_idx = idx_train
    dataset.valid_idx = idx_val
    dataset.test_idx = idx_test
    return dataset


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


def load_planetoid_dataset(name, no_feat_norm=False):
    # import pdb
    # pdb.set_trace()
    if not no_feat_norm:
        transform = T.NormalizeFeatures()
        torch_dataset = Planetoid(root=f'{DATAPATH}/Planetoid',
                                  name=name, transform=transform)
    else:
        torch_dataset = Planetoid(root=f'{DATAPATH}/Planetoid', name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes
    print(f"Num nodes: {num_nodes}")

    dataset = NCDataset(name)

    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset


def load_geom_gcn_dataset(name):
    # graph_adjacency_list_file_path = '../../data/geom-gcn/{}/out1_graph_edges.txt'.format(
    #     name)
    # graph_node_features_and_labels_file_path = '../../data/geom-gcn/{}/out1_node_feature_label.txt'.format(
    #     name)
    graph_adjacency_list_file_path = os.path.join(DATAPATH, 'geom-gcn/{}/out1_graph_edges.txt'.format(name))
    graph_node_features_and_labels_file_path = os.path.join(DATAPATH,
                                                            'geom-gcn/{}/out1_node_feature_label.txt'.format(name))

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if name == 'film':
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(
                    line[0]) not in graph_labels_dict)
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(
                    line[1].split(','), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])
    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(
                    line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(
                    line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = adj.tocoo().astype(np.float32)
    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    print(features.shape)

    def preprocess_features(feat):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(feat.sum(1))
        rowsum = (rowsum == 0) * 1 + rowsum
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        feat = r_mat_inv.dot(feat)
        return feat

    features = preprocess_features(features)

    edge_index = torch.from_numpy(
        np.vstack((adj.row, adj.col)).astype(np.int64))
    node_feat = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    num_nodes = node_feat.shape[0]
    print(f"Num nodes: {num_nodes}")

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = labels

    return dataset


def load_wiki_new(name, no_feat_norm=False):
    path = os.path.join(DATAPATH, f'wiki_new/{name}/{name}_filtered.npz')
    data = np.load(path)
    # lst=data.files
    # for item in lst:
    #     print(item)
    node_feat = data['node_features']  # unnormalized
    labels = data['node_labels']
    edges = data['edges']  # (E, 2)
    edge_index = edges.T

    if not no_feat_norm:
        node_feat = normalize_feat(node_feat)

    dataset = NCDataset(name)

    edge_index = torch.as_tensor(edge_index)
    node_feat = torch.as_tensor(node_feat)
    labels = torch.as_tensor(labels)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': node_feat.shape[0]}
    dataset.label = labels

    return dataset


def load_disease_dataset():
    object_to_idx = {}
    idx_counter = 0
    edges = []
    name = "disease_nc"
    dataset = NCDataset(name)

    with open(os.path.join(DATAPATH, 'hgcn_data', f'{name}', f"{name}.edges.csv"), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    features = sp.load_npz(os.path.join(DATAPATH, 'hgcn_data', f'{name}', "{}.feats.npz".format("disease_nc")))
    if sp.issparse(features):
        features = features.toarray()
    features = normalize_feat(features)
    features = torch.tensor(features, dtype=torch.float)

    labels = np.load(os.path.join(DATAPATH, 'hgcn_data', f'{name}', "{}.labels.npy".format("disease_nc")))
    val_prop, test_prop = 0.10, 0.60
    idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop)
    num_nodes = adj.shape[0]
    edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
    labels = torch.LongTensor(labels)
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': features,
                     'num_nodes': num_nodes}
    dataset.label = labels
    dataset.train_idx = idx_train
    dataset.valid_idx = idx_val
    dataset.test_idx = idx_test

    return dataset


def load_20news(n_remove=0):
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    import pickle as pkl

    if path.exists(DATAPATH + '20news/20news.pkl'):
        data = pkl.load(open(DATAPATH + '20news/20news.pkl', 'rb'))
    else:
        categories = ['alt.atheism',
                      'comp.sys.ibm.pc.hardware',
                      'misc.forsale',
                      'rec.autos',
                      'rec.sport.hockey',
                      'sci.crypt',
                      'sci.electronics',
                      'sci.med',
                      'sci.space',
                      'talk.politics.guns']
        data = fetch_20newsgroups(subset='all', categories=categories)
        # with open(data_dir + '20news/20news.pkl', 'wb') as f:
        #     pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)

    vectorizer = CountVectorizer(stop_words='english', min_df=0.05)
    X_counts = vectorizer.fit_transform(data.data).toarray()
    transformer = TfidfTransformer(smooth_idf=False)
    features = transformer.fit_transform(X_counts).todense()
    features = torch.Tensor(features)
    y = data.target
    y = torch.LongTensor(y)

    num_nodes = features.shape[0]

    if n_remove > 0:
        num_nodes -= n_remove
        features = features[:num_nodes, :]
        y = y[:num_nodes]

    dataset = NCDataset('20news')
    dataset.graph = {'edge_index': None,
                     'edge_feat': None,
                     'node_feat': features,
                     'num_nodes': num_nodes}
    dataset.label = torch.LongTensor(y)

    return dataset


def load_mini_imagenet():
    import pickle as pkl

    dataset = NCDataset('mini_imagenet')

    data = pkl.load(open(os.path.join(DATAPATH, 'mini_imagenet/mini_imagenet.pkl'), 'rb'))
    x_train = data['x_train']
    x_val = data['x_val']
    x_test = data['x_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']

    features = torch.cat((x_train, x_val, x_test), dim=0)
    labels = np.concatenate((y_train, y_val, y_test))
    num_nodes = features.shape[0]

    dataset.graph = {'edge_index': None,
                     'edge_feat': None,
                     'node_feat': features,
                     'num_nodes': num_nodes}
    dataset.label = torch.LongTensor(labels)
    return dataset


if __name__ == '__main__':
    # load_airport()
    # load_wikipedia('squirrel')
    # load_wiki_new('chameleon')
    pass
