import numpy as np
import networkx as nx
from tqdm import tqdm
import argparse
import copy
import os
import random
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import class_rand_splits, eval_acc, evaluate, load_fixed_splits, adj_mul
from dataset import load_nc_dataset
from logger import Logger
from parse import parse_method, parser_add_default_args, parser_add_main_args
from torch_geometric.utils import (add_self_loops, remove_self_loops,
                                   to_undirected)
from sklearn.neighbors import kneighbors_graph

def generate_connected_scale_free_graph(num_nodes):
    while True:
        graph = nx.scale_free_graph(num_nodes)
        undirected_graph = graph.to_undirected()
        if nx.is_connected(undirected_graph):
            return undirected_graph

def graph_distance_matrix(graph, nodes):
    """
    Compute the pairwise shortest path distance matrix for given nodes in a graph.
    """
    num_nodes = len(nodes)
    dist_matrix = np.full((num_nodes, num_nodes), np.inf)  # Initialize with infinity

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            try:
                dist = nx.shortest_path_length(graph, source=nodes[i], target=nodes[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
            except nx.NetworkXNoPath:
                continue  # Keep infinity if no path exists

    return dist_matrix

def delta_hyp(dismat):
    """
    Computes delta hyperbolicity value from distance matrix.
    Only considers finite distances in the calculations.
    """
    p = 0
    row = dismat[p, :][np.newaxis, :]
    col = dismat[:, p][:, np.newaxis]
    XY_p = 0.5 * (row + col - dismat)

    finite_mask = np.isfinite(XY_p)
    if not np.any(finite_mask):
        return np.nan  

    masked_XY_p = np.where(finite_mask, XY_p, np.nan)
    maxmin = np.nanmax(np.minimum(masked_XY_p[:, :, None], masked_XY_p[None, :, :]), axis=1)

    if not np.any(np.isfinite(maxmin)):
        return np.nan  

    return np.nanmax(maxmin - masked_XY_p)

def batched_delta_hyp(graph, nodes, n_tries=8, batch_size=1500):
    vals = []
    for _ in tqdm(range(n_tries)):
        idx = np.random.choice(len(nodes), batch_size, replace=False)
        node_batch = [nodes[i] for i in idx]
        distmat = graph_distance_matrix(graph, node_batch)

        finite_distmat = distmat[np.isfinite(distmat)]
        if finite_distmat.size == 0:
            continue  

        diam = np.max(finite_distmat)
        if diam == 0 or np.isnan(diam):
            continue  

        delta_rel = 2 * delta_hyp(distmat) / diam
        if not np.isnan(delta_rel):  # Only append valid values
            vals.append(delta_rel)

    if len(vals) == 0:
        return np.nan, np.nan  

    return np.mean(vals), np.std(vals)

# Number of nodes in the scale-free network
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
dataset = load_nc_dataset(args)
num_nodes = dataset.graph['num_nodes']
adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
edge_index = dataset.graph['edge_index']
edge_index.cpu().detach().numpy()
for edge in edge_index.T:
    i, j = edge
    adj_matrix[i, j] = 1
    adj_matrix[j, i] = 1  

results = []

for run in range(1):
    scale_free_graph = nx.from_numpy_array(adj_matrix)

    nodes = list(scale_free_graph.nodes)

    mean_delta, std_delta = batched_delta_hyp(scale_free_graph, nodes)

    results.append((mean_delta, std_delta))

mean_deltas = np.array([result[0] for result in results])
std_deltas = np.array([result[1] for result in results])

overall_mean_delta = np.mean(mean_deltas)
overall_std_delta = np.mean(std_deltas)

print(f"Overall mean relative delta hyperbolicity on the scale-free network: {overall_mean_delta}")
print(f"Overall standard deviation of relative delta hyperbolicity on the scale-free network: {overall_std_delta}")