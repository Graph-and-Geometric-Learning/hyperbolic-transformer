import torch
import torch.nn as nn
import torch.nn.functional as F

from hypformer import HypFormer

torch.manual_seed(42)

# Generate pseudo input data, this data is n by d
num_sample = 10
num_features = 16
num_classes = 5

# Generate random node features
x = torch.randn(num_sample, num_features)

# Define model parameters
in_channels = num_features
hidden_channels = 32
out_channels = num_classes

# Create an args object with necessary attributes
class Args:
    def __init__(self):
        self.k_in = 1.0
        self.k_out = 1.0
        self.decoder_type = 'hyp'
        self.device = 'cpu'
        self.add_positional_encoding = True
        self.attention_type = 'full'
        self.power_k = 2
        self.trans_heads_concat = False

args = Args()

# Instantiate the model
model = HypFormer(
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    trans_num_layers=2,
    trans_num_heads=4,
    trans_dropout=0.1,
    trans_use_bn=True,
    trans_use_residual=True,
    trans_use_weight=True,
    trans_use_act=True,
    args=args
)

# Forward pass
output = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
