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
from data_utils import class_rand_splits, eval_acc, evaluate, load_fixed_splits
from dataset import load_nc_dataset
from logger import Logger
from parse import parse_method, parser_add_main_args
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import (add_self_loops, remove_self_loops,
                                   to_undirected)
from manifolds.hyp_layer import Optimizer

warnings.filterwarnings('ignore')


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True


### Parse args ###
parser = argparse.ArgumentParser(description='Medium Data Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print('====' * 20)
print(args)
fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
    print('>> Using CPU')
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print('>> Using GPU: ' + str(args.device))

### Load and preprocess data ###
dataset = load_nc_dataset(args)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

dataset_name = args.dataset

if args.rand_split:
    print('>> loading random splits ...')
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    print('>> loading random class splits ...')
    split_idx_lst = [class_rand_splits(
        dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]
else:
    print('>> loading fixed splits ...')
    split_idx_lst = load_fixed_splits(
        dataset, name=args.dataset, protocol=args.protocol)

if args.dataset in ('mini', '20news'):
    adj_knn = kneighbors_graph(dataset.graph['node_feat'], n_neighbors=args.knn_num, include_self=True)
    edge_index = torch.tensor(adj_knn.nonzero(), dtype=torch.long)
    dataset.graph['edge_index'] = edge_index

n = dataset.graph['num_nodes']
num_class = max(dataset.label.max().item() + 1, dataset.label.shape[1])
num_class = (int)(num_class)
node_feat_dim = dataset.graph['node_feat'].shape[1]
args.in_channels = node_feat_dim
args.out_channels = num_class

dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device),
dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)

print(f">> num nodes {n} | num classes {num_class} | num node feats {node_feat_dim}")

if args.dataset in ('deezer-europe'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

eval_func = eval_acc

# ===============================================================================
logger = Logger(args.runs, args)
for run in range(args.runs):
    print(f'🔥Run {run + 1}/{args.runs}')
    if args.dataset in ['cora', 'citeseer', 'pubmed', 'airport', 'disease'] and args.protocol == 'semi':
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)  # get train split
    model = parse_method(args, device)  # load model
    optimizer = Optimizer(model, args)  # load optimizer

    best_val = float('-inf')
    patience = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        emb = None
        out = model(dataset)
        out = F.log_softmax(out, dim=1)
        loss = criterion(
            out[train_idx], dataset.label.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()

        result = evaluate(model, dataset, split_idx, eval_func, criterion, args)
        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            best_val = result[1]
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%')
    logger.print_statistics(run)
    # delete the model and optimizer and start a new run
    del model, optimizer

results = logger.print_statistics()
print(results)
out_folder = 'results'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)


def make_print(method):
    print_str = ''
    if args.rand_split_class:
        print_str += f'label per class:{args.label_num_per_class}, valid:{args.valid_num},test:{args.test_num}\n'
    else:
        print_str += f'method: {args.method} hidden: {args.hidden_channels} lr:{args.lr}\n'
    return print_str


if args.save_result:
    mkdirs(f'results/{args.dataset}')
    csvfilename = f'results/{args.dataset}/{args.dataset}_{args.method}_{args.hidden_channels}.csv'
    logger.save(vars(args), results, csvfilename)
