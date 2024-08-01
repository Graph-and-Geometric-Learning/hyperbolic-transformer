import argparse
import random
import time
import warnings
from datetime import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import (to_undirected, remove_self_loops, add_self_loops,
                                   subgraph, k_hop_subgraph)
import wandb

from logger import Logger
from dataset import load_dataset
from data_utils import (normalize, gen_normalized_adjs, eval_acc, eval_rocauc, eval_f1,
                        to_sparse_tensor, load_fixed_splits, adj_mul, compute_degrees)
from eval import evaluate_large, evaluate_batch
from parse import parse_method, parser_add_main_args
from manifolds import Optimizer

warnings.filterwarnings('ignore')


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description='General Training Pipeline')
    parser_add_main_args(parser)
    return parser.parse_args()


def get_device(use_cpu, device_id):
    if use_cpu:
        print('>> Using CPU (🐢🐢) ')
        return torch.device("cpu")
    else:
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        print('>> Using GPU (✈️✈️) ')
        return device


def load_and_preprocess_data(args):
    print(f'>> Loading dataset {args.dataset} (⏳⏳)')
    if 'hypformer_data_dir' in os.environ:
        args.data_dir = os.environ['hypformer_data_dir']
    dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset)

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    return dataset


def get_data_splits(args, dataset):
    if args.rand_split:
        split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                         for _ in range(args.runs)]
        print('>> Using random split ')
    elif args.rand_split_class:
        split_idx_lst = [dataset.get_idx_split(split_type='class', label_num_per_class=args.label_num_per_class)
                         for _ in range(args.runs)]
        print('>> Using random class split ')
    elif args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products', 'amazon2m', 'ogbn-papers100M',
                          'ogbn-papers100M-sub']:
        split_idx_lst = [dataset.load_fixed_splits()
                         for _ in range(args.runs)]
        print('>> Using fixed split ')
    else:
        split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset, protocol=args.protocol)
        print('>> Using fixed split ')

    return split_idx_lst


def print_dataset_info(dataset):
    n = dataset.graph['num_nodes']
    e = dataset.graph['edge_index'].shape[1]
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]

    print(f">> Dataset {dataset.name} | num nodes {n} | num edges {e} | num node feats {d} | num classes {c}")

    return n, c, d


def compute_and_print_degrees(dataset):
    degrees = compute_degrees(dataset.graph['edge_index'], dataset.graph['num_nodes'])
    print(f">> Total degree is {degrees.sum()}")
    # print(degrees)
    print(f">> Degree shape is {degrees.shape}")

    print(f">> Highest degree is {degrees.max().item()}")
    print(f">> Lowest degree is {degrees.min().item()}")

    sorted_degrees, _ = torch.sort(degrees)
    percentile_index = int(len(sorted_degrees) * 0.8)
    threshold = sorted_degrees[percentile_index]
    print(f'>> Mean degree: {degrees.float().mean().item():.2f}')
    print(f'>> Std degree: {degrees.float().std().item():.2f}')
    print(f'>> Number of nodes with degree 0: {(degrees == 0).sum().item()}')
    print(f'>> Threshold: {threshold:.2f}')

    return degrees, threshold


def select_loss_function(args):
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        criterion = nn.BCEWithLogitsLoss()
        print(f'>> Using BCEWithLogitsLoss for {args.dataset}')
    else:
        criterion = nn.NLLLoss()
        print(f'>> Using NLLLoss for {args.dataset}')

    return criterion


def select_eval_function(args):
    if args.metric == 'rocauc':
        eval_func = eval_rocauc
        print('>> Using ROC-AUC metric ')
    elif args.metric == 'f1':
        eval_func = eval_f1
        print('>> Using F1 metric ')
    else:
        eval_func = eval_acc
        print('>> Using Accuracy metric ')

    return eval_func


def preprocess_graph(dataset, args):
    dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
    dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=dataset.graph['num_nodes'])

    if not args.directed and args.dataset != 'ogbn-proteins':
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        print('>> Symmetrized the graph ')


def convert_labels_to_one_hot(dataset, args):
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            return F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
    return dataset.label


def initialize_wandb(args, run):
    if args.wandb_name == '0':
        now = datetime.now()
        timestamp = now.strftime("%m%d-%H%M")
        args.wandb_name = timestamp
        if args.use_wandb:
            wandb.init(project=f'HyperbolicFormer({args.dataset})', config=vars(args),
                   name=f'{args.dataset}-Params-{args.wandb_name}-run-{run}')

def train_and_evaluate(args, dataset, split_idx_lst, device, criterion, eval_func):
    n, c, d = print_dataset_info(dataset)
    degrees, threshold = compute_and_print_degrees(dataset)
    preprocess_graph(dataset, args)
    true_label = convert_labels_to_one_hot(dataset, args)
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        initialize_wandb(args, run)

        split_idx = split_idx_lst[run] if args.dataset not in ['cora', 'citeseer',
                                                               'pubmed'] or args.protocol != 'semi' else split_idx_lst[
            0]
        train_mask = torch.zeros(n, dtype=torch.bool)
        train_mask[split_idx['train']] = True

        model = parse_method(args, c, d, device)
        model.reset_parameters()
        optimizer = Optimizer(model, args)

        for epoch in range(args.epochs):
            loss = train_one_epoch(epoch, args, dataset, device, model, optimizer, criterion, n, train_mask, true_label, degrees,
                                   threshold, eval_func, split_idx, logger, run)

        logger.print_statistics(run)
        if args.use_wandb:
            wandb.finish()

    results = logger.print_statistics()
    logger.save(vars(args), results, f'results/{args.dataset}.csv')


def train_one_epoch(epoch, args, dataset, device, model, optimizer, criterion, n, train_mask, true_label, degrees, threshold,
                    eval_func, split_idx, logger, run):
    model.to(device)
    model.train()
    train_start = time.time()
    idx = torch.randperm(n)
    num_batch = n // args.batch_size + (n % args.batch_size > 0)

    for i in range(num_batch):
        batch_data = prepare_batch_data(i, args,dataset, idx, n, train_mask, true_label, degrees, device)
        optimizer.zero_grad()
        out_i = model(batch_data['x_i'], batch_data['edge_index_i'])
        loss = compute_loss(args, criterion, out_i, batch_data['train_mask_i'], batch_data['y_i'])
        loss.backward()
        optimizer.step()

    print(f'🔥🔥 Epoch: {epoch:02d}, Loss: {loss:.4f} || Train Time: {time.time() - train_start:.2f}s')
    evaluate_epoch(epoch, args, model, dataset, split_idx, eval_func, criterion, degrees, threshold, device, logger,
                   run, loss, true_label)

    return loss


def prepare_batch_data(i, args, dataset, idx, n, train_mask, true_label, degrees, device):
    idx_i = idx[i * args.batch_size:(i + 1) * args.batch_size]
    train_mask_i = train_mask[idx_i]
    x_i = dataset.graph['node_feat'][idx_i].to(device)
    edge_index_i, _ = subgraph(idx_i, dataset.graph['edge_index'], num_nodes=n, relabel_nodes=True)
    edge_index_i = edge_index_i.to(device)
    y_i = true_label[idx_i].to(device)

    return {'idx_i': idx_i, 'train_mask_i': train_mask_i, 'x_i': x_i, 'edge_index_i': edge_index_i, 'y_i': y_i}


def compute_loss(args, criterion, out_i, train_mask_i, y_i):
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        return criterion(out_i[train_mask_i], y_i.squeeze(1)[train_mask_i].to(torch.float))
    else:
        out_i = F.log_softmax(out_i, dim=1)
        return criterion(out_i[train_mask_i], y_i.squeeze(1)[train_mask_i])


def evaluate_epoch(epoch, args, model, dataset, split_idx, eval_func, criterion, degrees, threshold, device, logger,
                   run, loss, true_label):
    if (epoch + 1) % args.eval_step == 0:
        if args.dataset == 'ogbn-papers100M':
            result = evaluate_batch(model, dataset, split_idx, args, device, dataset.graph['num_nodes'], true_label)
        else:
            result = evaluate_large(model, dataset, split_idx, eval_func, criterion, args, degrees, threshold,
                                    device=device)
        logger.add_result(run, result)

        if epoch % args.display_step == 0:
            display_evaluation_results(epoch, result, split_idx, degrees, threshold, loss)

        if args.use_wandb:
            wandb.log({"run": run, "epoch": epoch, "loss": loss.item(), "train_acc": result[0],
                       "val_acc": result[1], "test_acc": result[2], "val_loss": result[3]})


def display_evaluation_results(epoch, result, split_idx, degrees, threshold, loss):
    degrees_in_test = degrees[split_idx['test']]
    top_indices = split_idx['test'][degrees_in_test > threshold]
    bottom_indices = split_idx['test'][degrees_in_test <= threshold]

    max_top_acc = top_indices.shape[0] / split_idx['test'].shape[0]
    max_bottom_acc = bottom_indices.shape[0] / split_idx['test'].shape[0]
    print_str = (f'👉Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * result[0]:.2f}%, '
                 f'Valid: {100 * result[1]:.2f}%, Test: {100 * result[2]:.2f}%,  '
                 f'Top: {100 * result[5]:.2f} | {100 * max_top_acc:.2f}%,  '
                 f'Bottom: {100 * result[6]:.2f} | {100 * max_bottom_acc:.2f}%')
    print(print_str)


def main():
    args = parse_args()
    print('===' * 40)
    print('⚙️,', args)

    fix_seed(args.seed)
    device = get_device(args.cpu, args.device)
    dataset = load_and_preprocess_data(args)
    split_idx_lst = get_data_splits(args, dataset)

    criterion = select_loss_function(args)
    eval_func = select_eval_function(args)

    train_and_evaluate(args, dataset, split_idx_lst, device, criterion, eval_func)


if __name__ == "__main__":
    main()
