from hypformer import HypFormer


def parse_method(args, device):
    if args.method == 'hypformer':
        model = HypFormer(args=args).to(device)
    else:
        raise ValueError(f'Invalid method {args.method}')
    return model


def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--data_dir', type=str, default='../data', help='location of the data')
    parser.add_argument('--dataset', type=str, default='cora', help='name of dataset')
    parser.add_argument('--sub_dataset', type=str, default='gcn_data', help='name of sub dataset')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', type=int, default=0, help='use CPU instead of GPU')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1, help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.5, help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25, help='validation label proportion')
    parser.add_argument('--protocol', type=str, default='semi', help='protocol for cora datasets, semi or supervised')
    parser.add_argument('--rand_split', type=int, default=0, help='use random splits')
    parser.add_argument('--rand_split_class', type=int, default=0,
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class (randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500, help='total number of validation nodes')
    parser.add_argument('--test_num', type=int, default=500, help='total number of test nodes')
    parser.add_argument('--no_feat_norm', type=int, default=1)

    # display and utility
    parser.add_argument('--display_step', type=int, default=50, help='how often to print')

    # model
    parser.add_argument('--method', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=32)

    # gnn branch
    parser.add_argument('--use_graph', type=int, default=1, help='use graph encoder or not')
    parser.add_argument('--graph_weight', type=float, default=0.5, help='weight for graph encoder')
    parser.add_argument('--gnn_use_bn', type=int, default=1, help='use batchnorm for each GNN layer or not')
    parser.add_argument('--gnn_use_residual', type=int, default=1, help='use residual link for each GNN layer or not')
    parser.add_argument('--gnn_use_weight', type=int, default=0, help='use weight for GNN convolution')
    parser.add_argument('--gnn_use_init', type=int, default=0, help='use initial feat for each GNN layer or not')
    parser.add_argument('--gnn_use_act', type=int, default=1, help='use activation for each GNN layer or not')
    parser.add_argument('--gnn_num_layers', type=int, default=2, help='number of layers for GNN')
    parser.add_argument('--gnn_dropout', type=float, default=0.5)
    parser.add_argument('--knn_num', type=int, default=5, help='number of k for KNN graph')

    # attention (Transformer) branch
    parser.add_argument('--trans_num_heads', type=int, default=1, help='number of attention heads')
    parser.add_argument('--trans_heads_concat', type=int, default=0, help='concatenate multi-head attentions or not')
    parser.add_argument('--trans_use_weight', type=int, default=1, help='use weight for transformer convolution or not')
    parser.add_argument('--trans_use_bn', type=int, default=0, help='use batchnorm for each transformer layer or not')
    parser.add_argument('--trans_use_residual', type=int, default=0,
                        help='use residual link for each transformer layer or not')
    parser.add_argument('--trans_use_act', type=int, default=0, help='use activation for each transformer layer or not')
    parser.add_argument('--trans_num_layers', type=int, default=1, help='number of layers for all-pair attention')

    parser.add_argument('--trans_dropout', type=float, default=0.0, help='transformer dropout')
    parser.add_argument('--k_in', type=float, default=1.0, help='manifold_in curvature')
    parser.add_argument('--k_hidden', type=float, default=1.0, help='Curvature for input layer (default: 1.0)')
    parser.add_argument('--k_out', type=float, default=1.0, help='manifold_out curvature')
    parser.add_argument('--power_k', type=float, default=2.0, help='power k for query and key')
    parser.add_argument('--attention_type', type=str, default='linear_focused',
                        help='attention type: linear_focused, or full')
    parser.add_argument('--add_positional_encoding', type=int, default=1, help='add positional encoding or not')
    # training
    parser.add_argument('--patience', type=int, default=200, help='early stopping patience')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hyp_lr', type=float, default=0.01)

    parser.add_argument('--optimizer_type', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--hyp_optimizer_type', type=str, default='radam', choices=['radam', 'rsgd'])
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--hyp_weight_decay', type=float, default=0.005)

    parser.add_argument('--decoder_type', type=str, default='euc')

    parser.add_argument('--use_wandb', type=int, default=0, help='use wandb for logging')
    parser.add_argument('--wandb_name', type=int, default=0, help='wandb run name')
    parser.add_argument('--run_id', type=str, default='0', help='Run ID (default: 0)')
    parser.add_argument('--save_result', type=int, default=0, help='save whole test result')
