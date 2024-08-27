import pdb
import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from manifolds.hyp_layer import HypLinear, HypLayerNorm, HypActivation, HypDropout, HypCLS
from manifolds.lorentz import Lorentz
from geoopt import ManifoldParameter
from gnns import GraphConv, GCN


class TransConvLayer(nn.Module):
    def __init__(self, manifold, in_channels, out_channels, num_heads, use_weight=True, args=None):
        """
        Initializes a TransConvLayer instance.

        Args:
            manifold: The manifold to use for the layer.
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            num_heads: The number of attention heads.
            use_weight: Whether to use weights for the attention mechanism. Defaults to True.
            args: Additional arguments for the layer, including attention_type, power_k, and trans_heads_concat.

        Returns:
            None
        """
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight
        self.attention_type = args.attention_type

        self.Wk = nn.ModuleList()
        self.Wq = nn.ModuleList()
        for i in range(self.num_heads):
            self.Wk.append(HypLinear(self.manifold, self.in_channels, self.out_channels))
            self.Wq.append(HypLinear(self.manifold, self.in_channels, self.out_channels))

        if use_weight:
            self.Wv = nn.ModuleList()
            for i in range(self.num_heads):
                self.Wv.append(HypLinear(self.manifold, in_channels, out_channels))

        self.scale = nn.Parameter(torch.tensor([math.sqrt(out_channels)], requires_grad=True))
        self.bias = nn.Parameter(torch.zeros(()))
        self.norm_scale = nn.Parameter(torch.ones(()))
        self.v_map_mlp = nn.Linear(in_channels, out_channels, bias=True)
        self.power_k = args.power_k
        self.trans_heads_concat = args.trans_heads_concat

        if self.trans_heads_concat:
            self.final_linear = nn.Linear(out_channels * self.num_heads, out_channels, bias=True)

    def full_attention(self, qs, ks, vs, output_attn=False):
        # normalize input
        # qs = HypNormalization(self.manifold)(qs)
        # ks = HypNormalization(self.manifold)(ks)

        # negative squared distance (less than 0)
        att_weight = 2 + 2 * self.manifold.cinner(qs.transpose(0, 1), ks.transpose(0, 1))  # [H, N, N]
        att_weight = att_weight / self.scale + self.bias  # [H, N, N]

        att_weight = nn.Softmax(dim=-1)(att_weight)  # [H, N, N]
        att_output = self.manifold.mid_point(vs.transpose(0, 1), att_weight)  # [N, H, D]
        att_output = att_output.transpose(0, 1)  # [N, H, D]

        att_output = self.manifold.mid_point(att_output)
        if output_attn:
            return att_output, att_weight
        else:
            return att_output

    @staticmethod
    def fp(x, p=2):
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm_x_p = torch.norm(x ** p, p=2, dim=-1, keepdim=True)
        return (norm_x / norm_x_p) * x ** p

    def linear_focus_attention(self, hyp_qs, hyp_ks, hyp_vs, output_attn=False):
        qs = hyp_qs[..., 1:]
        ks = hyp_ks[..., 1:]
        v = hyp_vs[..., 1:]

        phi_qs = (F.relu(qs) + 1e-6) / self.norm_scale.abs()  # [N, H, D]
        phi_ks = (F.relu(ks) + 1e-6) / self.norm_scale.abs()  # [N, H, D]
        # v = (F.relu(v) + 1e-6) / self.norm_scale.abs()

        phi_qs = self.fp(phi_qs, p=self.power_k)  # [N, H, D]
        phi_ks = self.fp(phi_ks, p=self.power_k)  # [N, H, D]

        # Step 1: Compute the kernel-transformed sum of K^T V across all N for each head
        k_transpose_v = torch.einsum('nhm,nhd->hmd', phi_ks, v)  # [H, D, D] 

        # Step 2: Compute the kernel-transformed dot product of Q with the above result
        numerator = torch.einsum('nhm,hmd->nhd', phi_qs, k_transpose_v)  # [N, H, D]

        # Step 3: Compute the normalizing factor as the kernel-transformed sum of K
        denominator = torch.einsum('nhd,hd->nh', phi_qs, torch.einsum('nhd->hd', phi_ks))  # [N, H]
        denominator = denominator.unsqueeze(-1)  # [N, H, D] for broadcasting

        # Step 4: Normalize the numerator with the denominator
        attn_output = numerator / (denominator + 1e-6)  # [N, H, D]

        # Map vs through v_map_mlp and ensure it is the correct shape
        vss = self.v_map_mlp(v)  # [N, H, D]
        attn_output = attn_output + vss  # preserve its rank, [N, H, D]

        if self.trans_heads_concat:
            attn_output = self.final_linear(attn_output.reshape(-1, self.num_heads * self.out_channels))
        else:
            attn_output = attn_output.mean(dim=1)

        attn_output_time = ((attn_output ** 2).sum(dim=-1, keepdims=True) + self.manifold.k) ** 0.5
        attn_output = torch.cat([attn_output_time, attn_output], dim=-1)


        if output_attn:
            # Calculate attention weights
            attention = torch.einsum('nhd,mhd->nmh', phi_qs, phi_ks)  # [N, M, H]
            attention = attention / (denominator + 1e-6)  # Normalize

            # Average attention across heads if needed
            attention = attention.mean(dim=-1)  # [N, M]
            return attn_output, attention
        else:
            return attn_output

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
        # feature transformation
        q_list = []
        k_list = []
        v_list = []
        for i in range(self.num_heads):
            q_list.append(self.Wq[i](query_input))
            k_list.append(self.Wk[i](source_input))
            if self.use_weight:
                v_list.append(self.Wv[i](source_input))
            else:
                v_list.append(source_input)

        query = torch.stack(q_list, dim=1)  # [N, H, D]
        key = torch.stack(k_list, dim=1)  # [N, H, D]
        value = torch.stack(v_list, dim=1)  # [N, H, D]

        if output_attn:
            if self.attention_type == 'linear_focused':
                attention_output, attn = self.linear_focus_attention(
                    query, key, value, output_attn)  # [N, H, D]
            elif self.attention_type == 'full':
                attention_output, attn = self.full_attention(
                    query, key, value, output_attn)
            else:
                raise NotImplementedError
        else:
            if self.attention_type == 'linear_focused':
                attention_output = self.linear_focus_attention(
                    query, key, value)  # [N, H, D]
            elif self.attention_type == 'full':
                attention_output = self.full_attention(
                    query, key, value)  # [N, H, D]

        final_output = attention_output

        if output_attn:
            return final_output, attn
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, manifold_in, manifold_hidden, manifold_out, in_channels, hidden_channels, args=None):
        super().__init__()
        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = args.trans_num_layers
        self.num_heads = args.trans_num_heads
        self.dropout_rate = args.trans_dropout
        self.use_bn = args.trans_use_bn
        self.residual = args.trans_use_residual
        self.use_act = args.trans_use_act
        self.use_weight = args.trans_use_weight

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.fcs.append(HypLinear(self.manifold_in, self.in_channels, self.hidden_channels, self.manifold_hidden))
        self.bns.append(HypLayerNorm(self.manifold_hidden, self.hidden_channels))

        self.add_pos_enc = args.add_positional_encoding
        self.positional_encoding = HypLinear(self.manifold_in, self.in_channels, self.hidden_channels,
                                             self.manifold_hidden)
        self.epsilon = torch.tensor([1.0], device=args.device)

        for i in range(self.num_layers):
            self.convs.append(
                TransConvLayer(self.manifold_hidden, self.hidden_channels, self.hidden_channels,
                               num_heads=self.num_heads,
                               use_weight=self.use_weight, args=args))
            self.bns.append(HypLayerNorm(self.manifold_hidden, self.hidden_channels))

        self.dropout = HypDropout(self.manifold_hidden, self.dropout_rate)
        self.activation = HypActivation(self.manifold_hidden, activation=F.relu)

        self.fcs.append(HypLinear(self.manifold_hidden, self.hidden_channels, self.hidden_channels, self.manifold_out))

    def forward(self, x_input):
        layer_ = []

        # the original inputs are in Euclidean
        x = self.fcs[0](x_input, x_manifold='euc')
        # add positional embedding
        if self.add_pos_enc:
            x_pos = self.positional_encoding(x_input, x_manifold='euc')
            x = self.manifold_in.mid_point(torch.stack((x, self.epsilon * x_pos), dim=1))

        if self.use_bn:
            x = self.bns[0](x)
        if self.use_act:
            x = self.activation(x)
        if self.dropout_rate > 0:
            x = self.dropout(x, training=self.training)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, x)
            if self.residual:
                x = self.manifold_in.mid_point(torch.stack((x, layer_[i]), dim=1))
            if self.use_bn:
                x = self.bns[i + 1](x)
            # if self.use_act:
            #     x = self.activation(x)
            # if self.dropout_rate > 0:
            #     x = self.dropout(x, training=self.training)
            layer_.append(x)

        x = self.fcs[-1](x)
        return x

    def get_attentions(self, x_input):
        layer_, attentions = [], []

        # the original inputs are in Euclidean
        x = self.fcs[0](x_input, x_manifold='euc')
        # add positional embedding
        if self.add_pos_enc:
            x_pos = self.positional_encoding(x_input, x_manifold='euc')
            x = self.manifold_in.mid_point(torch.stack((x, self.epsilon * x_pos), dim=1))

        if self.use_bn:
            x = self.bns[0](x)
        if self.use_act:
            x = self.activation(x)
        if self.dropout_rate > 0:
            x = self.dropout(x, training=self.training)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.manifold_in.mid_point(torch.stack((x, layer_[i]), dim=1))
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]


class HypFormer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.manifold_in = Lorentz(k=float(args.k_in))
        self.manifold_hidden = Lorentz(k=float(args.k_out))
        self.manifold_out = Lorentz(k=float(args.k_out))

        self.in_channels = args.in_channels
        self.hidden_channels = args.hidden_channels
        self.out_channels = args.out_channels
        self.use_graph = args.use_graph
        self.graph_weight = args.graph_weight

        # self.aggregate_type = args.aggregate_type
        self.decoder_type = args.decoder_type

        self.trans_conv = TransConv(self.manifold_in, self.manifold_hidden, self.manifold_out, self.in_channels,
                                    self.hidden_channels, args=args)
        # self.graph_conv = GCN(self.in_channels, self.hidden_channels, self.out_channels, args=args) if args.use_graph else None
        self.graph_conv = GraphConv(self.in_channels, self.hidden_channels, args=args) if self.use_graph else None

        if self.decoder_type == 'euc':
            self.decode_trans = nn.Linear(self.hidden_channels, self.out_channels)
            self.decode_graph = nn.Linear(self.hidden_channels, self.out_channels)
        elif self.decoder_type == 'hyp':
            self.decode_graph = HypLinear(self.manifold_out, self.hidden_channels, self.hidden_channels)
            self.decode_trans = HypCLS(self.manifold_out, self.hidden_channels, self.out_channels)
        else:
            raise NotImplementedError

    def forward(self, dataset):
        x, edge_index = dataset.graph['node_feat'], dataset.graph['edge_index'][0]
        x1 = self.trans_conv(x)  # hyperbolic Transformer encoder

        if self.use_graph:
            x2 = self.graph_conv(x, edge_index)  # Graph encoder
            if self.decoder_type == 'euc':
                x = (1 - self.graph_weight) * self.decode_trans(
                    self.manifold_out.logmap0(x1)[..., 1:]) + self.graph_weight * self.decode_graph(x2)
            elif self.decoder_type == 'hyp':
                z_graph_hyp = self.decode_graph(x2, x_manifold='euc')
                z_hyp = torch.stack([(1 - self.graph_weight) * x1, self.graph_weight * z_graph_hyp], dim=1)
                z = self.manifold_out.mid_point(z_hyp)
                x = self.decode_trans(z)
            else:
                raise NotImplementedError
        else:
            if self.decoder_type == 'euc':
                x = self.decode_trans(self.manifold_out.logmap0(x1)[..., 1:])
            elif self.decoder_type == 'hyp':
                x = self.decode_trans(x1)
            else:
                raise NotImplementedError
        return x

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]
        return attns
