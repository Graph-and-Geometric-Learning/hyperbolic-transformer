
from hypformer import HypFormer

def parse_method(args, c, d, device):
    model = HypFormer(d, args.hidden_channels, c, graph_weight=args.graph_weight, aggregate=args.aggregate,
                    trans_num_layers=args.trans_num_layers, trans_dropout=args.trans_dropout, trans_num_heads=args.trans_num_heads,
                      trans_use_bn=args.trans_use_bn, trans_use_residual=args.trans_use_residual, trans_use_weight=args.trans_use_weight, trans_use_act=args.trans_use_act,
                     gnn_num_layers=args.gnn_num_layers, gnn_dropout=args.gnn_dropout, gnn_use_bn=args.gnn_use_bn,
                      gnn_use_residual=args.gnn_use_residual, gnn_use_weight=args.gnn_use_weight, gnn_use_init=args.gnn_use_init, gnn_use_act=args.gnn_use_act,
                     args=args).to(device)
    return model



def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='proteins', help='Name of the dataset to be used (default: proteins)')
    parser.add_argument('--sub_dataset', type=str, default='', help='Sub-dataset to be used (if any)')
    parser.add_argument('--data_dir', type=str, default='../data', help='Directory where the data is stored')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID to be used (default: 0)')
    parser.add_argument('--seed', type=int, default=123, help='Random seed for reproducibility (default: 123)')
    parser.add_argument('--cpu', type=int, choices=[0, 1], default=0, help='Use CPU instead of GPU (0: False, 1: True)')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs (default: 500)')
    parser.add_argument('--runs', type=int, default=1, help='Number of distinct runs (default: 1)')
    parser.add_argument('--directed', type=int, choices=[0, 1], default=0, help='Set to use directed graph (0: False, 1: True)')
    parser.add_argument('--train_prop', type=float, default=.5, help='Proportion of training labels (default: 0.5)')
    parser.add_argument('--valid_prop', type=float, default=.25, help='Proportion of validation labels (default: 0.25)')
    parser.add_argument('--protocol', type=str, default='semi',
                        help='Protocol for cora datasets: semi or supervised (default: semi)')
    parser.add_argument('--rand_split', type=int, choices=[0, 1], help='Use random splits (0: False, 1: True)')
    parser.add_argument('--rand_split_class', type=int, choices=[0, 1],
                        help='Use random splits with a fixed number of labeled nodes per class (0: False, 1: True)')
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='Number of labeled nodes per class (default: 20)')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'f1'],
                        help='Evaluation metric (default: acc)')

    parser.add_argument('--use_graph', type=int, choices=[0, 1], help='Use input graph (0: False, 1: True)')
    parser.add_argument('--aggregate', type=str, default='add', help='Aggregate type: add or cat (default: add)')
    parser.add_argument('--graph_weight', type=float, default=0.8, help='Weight for the graph (default: 0.8)')
    parser.add_argument('--gnn_use_bn', type=int, choices=[0, 1],
                        help='Use batch normalization in each GNN layer (0: False, 1: True)')
    parser.add_argument('--gnn_use_residual', type=int, choices=[0, 1],
                        help='Use residual connections in each GNN layer (0: False, 1: True)')
    parser.add_argument('--gnn_use_weight', type=int, choices=[0, 1], help='Use weight for GNN convolution (0: False, 1: True)')
    parser.add_argument('--gnn_use_init', type=int, choices=[0, 1],
                        help='Use initial features in each GNN layer (0: False, 1: True)')
    parser.add_argument('--gnn_use_act', type=int, choices=[0, 1], help='Use activation function in each GNN layer (0: False, 1: True)')
    parser.add_argument('--gnn_num_layers', type=int, default=2, help='Number of GNN layers (default: 2)')
    parser.add_argument('--gnn_dropout', type=float, default=0.0, help='Dropout rate for GNN layers (default: 0.0)')

    # all-pair attention (Transformer) branch
    parser.add_argument('--method', type=str, default='hypformer', help='method to be used (default: hypformer)')
    parser.add_argument('--hidden_channels', type=int, default=32, help='Number of hidden channels (default: 32)')
    parser.add_argument('--trans_num_heads', type=int, default=1,
                        help='Number of heads for attention in Transformer (default: 1)')
    parser.add_argument('--trans_use_weight', type=int, choices=[0, 1],
                        help='Use weight for Transformer convolution (0: False, 1: True)')
    parser.add_argument('--trans_use_bn', type=int, choices=[0, 1],
                        help='Use layer normalization in Transformer (0: False, 1: True)')
    parser.add_argument('--trans_use_residual', type=int, choices=[0, 1],
                        help='Use residual connections in each Transformer layer (0: False, 1: True)')
    parser.add_argument('--trans_use_act', type=int, choices=[0, 1],
                        help='Use activation function in each Transformer layer (0: False, 1: True)')
    parser.add_argument('--trans_num_layers', type=int, default=2, help='Number of Transformer layers (default: 2)')
    parser.add_argument('--trans_dropout', type=float, help='Dropout rate for Transformer layers')
    parser.add_argument('--add_positional_encoding', type=int, default=1,
                        help='Add positional encoding to Transformer layers (default: 1)')

    # display and utility
    parser.add_argument('--display_step', type=int, default=1, help='Frequency of display updates (default: 1)')
    parser.add_argument('--eval_step', type=int, default=1, help='Frequency of evaluation steps (default: 1)')
    parser.add_argument('--cached', type=int, choices=[0, 1], help='Use cached data for faster processing (0: False, 1: True)')
    parser.add_argument('--print_prop', type=int, choices=[0, 1],
                        help='Print proportions of predicted classes (0: False, 1: True)')
    parser.add_argument('--save_result', type=int, choices=[0, 1], default=0, help='Save the result of the run (0: False, 1: True)')
    parser.add_argument('--save_model', type=int, choices=[0, 1], help='Save the model after training (0: False, 1: True)')
    parser.add_argument('--use_pretrained', type=int, choices=[0, 1], help='Use a pre-trained model (0: False, 1: True)')
    parser.add_argument('--save_att', type=int, choices=[0, 1],
                        help='Save attention weights for visualization (0: False, 1: True)')
    parser.add_argument('--model_dir', type=str, default='checkpoints/',
                        help='Directory to save the model checkpoints (default: checkpoints/)')

    # other gnn parameters (for baselines)
    parser.add_argument('--hops', type=int, default=2, help='Number of hops for SGC (default: 2)')
    parser.add_argument('--gat_heads', type=int, default=4, help='Number of attention heads for GAT (default: 4)')
    parser.add_argument('--out_heads', type=int, default=1, help='Number of output heads for GAT (default: 1)')

    # training
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (default: 0.0)')
    parser.add_argument('--hyp_weight_decay', type=float, default=0.005,
                        help='Weight decay for Hyperbolic space (default: 0.005)')

    parser.add_argument('--optimizer_type', type=str, default='adam', choices=['adam', 'sgd'],
                        help='Optimizer type for Euclidean space (default: adam)')
    parser.add_argument('--hyp_optimizer_type', type=str, default='radam', choices=['radam', 'rsgd'],
                        help='Optimizer type for Hyperbolic space (default: radam)')

    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--hyp_lr', type=float, default=0.01, help='Hyperbolic learning rate (default: 0.01)')

    parser.add_argument('--batch_size', type=int, default=10000,
                        help='Mini-batch size for training large graphs (default: 10000)')
    parser.add_argument('--patience', type=int, default=200, help='Early stopping patience (default: 200)')
    parser.add_argument('--k_in', type=float, default=1.0, help='Curvature for input layer (default: 1.0)')
    parser.add_argument('--k_hidden', type=float, default=1.0, help='Curvature for hidden layer (default: 1.0)')
    parser.add_argument('--k_out', type=float, default=1.0, help='Curvature for output layer (default: 1.0)')
    parser.add_argument('--use_wandb', type=int, choices=[0, 1], help='Use Weights and Biases for logging (0: False, 1: True)')
    parser.add_argument('--wandb_name', type=str, default='0', help='Weights and Biases project name (default: 0)')
    parser.add_argument('--power_k', type=float, default=2.0, help='Power k for query and key (default: 2.0)')
    parser.add_argument('--attention_type', type=str, default='linear_focused',
                        help='Attention type: linear_focused, or full (default: linear_focused)')
    parser.add_argument('--run_id', type=str, default='0', help='Run ID (default: 0)')
    parser.add_argument('--save_whole_test_result', type=int, default=1, help='Save whole test result (default: 1)')
    parser.add_argument('--decoder_type', type=str, default='euc', help='Decoder type (default: euc)')
    parser.add_argument('--trans_heads_concat', type=int, default=0, help='Use heads concatenation for Transformer (default: 1)')



