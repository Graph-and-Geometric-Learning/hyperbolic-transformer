import pdb
import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from manifolds.layer import HypLinear, HypLayerNorm, HypActivation, HypDropout, HypNormalization, HypCLS
from manifolds.lorentz import Lorentz
from geoopt import ManifoldParameter

class TransConvLayer(nn.Module):
    def __init__(self, manifold, in_channels, out_channels, num_heads, use_weight=True, args=None):
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

        self.scale = nn.Parameter(torch.tensor([math.sqrt(out_channels)]))
        self.bias = nn.Parameter(torch.zeros(()))
        self.norm_scale = nn.Parameter(torch.ones(()))
        self.v_map_mlp = nn.Linear(in_channels, out_channels, bias=True)
        self.power_k = args.power_k
        self.trans_heads_concat = args.trans_heads_concat


    @staticmethod
    def fp(x, p=2):
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm_x_p = torch.norm(x ** p, p=2, dim=-1, keepdim=True)
        return (norm_x / norm_x_p) * x ** p

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

    def linear_focus_attention(self, hyp_qs, hyp_ks, hyp_vs, output_attn=False):
            qs = hyp_qs[..., 1:]
            ks = hyp_ks[..., 1:]
            v = hyp_vs[..., 1:]
            phi_qs = (F.relu(qs) + 1e-6) / (self.norm_scale.abs() + 1e-6)  # [N, H, D]
            phi_ks = (F.relu(ks) + 1e-6) / (self.norm_scale.abs() + 1e-6)  # [N, H, D]

            phi_qs = self.fp(phi_qs, p=self.power_k)  # [N, H, D]
            phi_ks = self.fp(phi_ks, p=self.power_k)  # [N, H, D]

            # Step 1: Compute the kernel-transformed sum of K^T V across all N for each head
            k_transpose_v = torch.einsum('nhm,nhd->hmd', phi_ks, v)  # [H, D, D]

            # Step 2: Compute the kernel-transformed dot product of Q with the above result
            numerator = torch.einsum('nhm,hmd->nhd', phi_qs, k_transpose_v)  # [N, H, D]

            # Step 3: Compute the normalizing factor as the kernel-transformed sum of K
            denominator = torch.einsum('nhd,hd->nh', phi_qs, torch.einsum('nhd->hd', phi_ks))  # [N, H]
            denominator = denominator.unsqueeze(-1)  #

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
                return attn_output, attn_output
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
                    query, key, value)
            else:
                raise NotImplementedError


        final_output = attention_output
        # multi-head attention aggregation
        # final_output = self.manifold.mid_point(final_output)

        if output_attn:
            return final_output, attn
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, manifold_in, manifold_hidden, manifold_out, in_channels, hidden_channels, num_layers=2, num_heads=1,
                 dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=True, args=None):
        super().__init__()
        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.use_bn = use_bn
        self.residual = use_residual
        self.use_act = use_act
        self.use_weight = use_weight

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.fcs.append(HypLinear(self.manifold_in, self.in_channels, self.hidden_channels, self.manifold_hidden))
        self.bns.append(HypLayerNorm(self.manifold_hidden, self.hidden_channels))

        self.add_pos_enc = args.add_positional_encoding
        self.positional_encoding = HypLinear(self.manifold_in, self.in_channels, self.hidden_channels, self.manifold_hidden)
        self.epsilon = torch.tensor([1.0], device=args.device)

        for i in range(self.num_layers):
            self.convs.append(
                TransConvLayer(self.manifold_hidden, self.hidden_channels, self.hidden_channels, num_heads=self.num_heads, use_weight=self.use_weight, args=args))
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
            x = self.manifold_hidden.mid_point(torch.stack((x, self.epsilon*x_pos), dim=1))

        if self.use_bn:
            x = self.bns[0](x)
        if self.use_act:
            x = self.activation(x)
        x = self.dropout(x, training=self.training)
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, x)
            if self.residual:
                x = self.manifold_hidden.mid_point(torch.stack((x, layer_[i]), dim=1))
            if self.use_bn:
                x = self.bns[i + 1](x)
            # if self.use_act:
            #     x = self.activation(x)
            # # x = self.dropout(x, training=self.training)
            layer_.append(x)

        x = self.fcs[-1](x)
        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.manifold_hidden.mid_point(torch.stack((x, layer_[i]), dim=1))
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]

class HypFormer(nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels,
                 trans_num_layers=1, trans_num_heads=1, trans_dropout=0.5, trans_use_bn=True, trans_use_residual=True,
                 trans_use_weight=True, trans_use_act=True,
                 args=None):
        """
        Initializes a HypFormer object.

        Args:
            in_channels (int): The number of input channels.
            hidden_channels (int): The number of hidden channels.
            out_channels (int): The number of output channels.
            trans_num_layers (int, optional): The number of layers in the TransConv module. Defaults to 1.
            trans_num_heads (int, optional): The number of attention heads in the TransConv module. Defaults to 1.
            trans_dropout (float, optional): The dropout rate in the TransConv module. Defaults to 0.5.
            trans_use_bn (bool, optional): Whether to use batch normalization in the TransConv module. Defaults to True.
            trans_use_residual (bool, optional): Whether to use residual connections in the TransConv module. Defaults to True.
            trans_use_weight (bool, optional): Whether to use learnable weights in the TransConv module. Defaults to True.
            trans_use_act (bool, optional): Whether to use activation functions in the TransConv module. Defaults to True.
            args (optional): Additional arguments.

        Raises:
            NotImplementedError: If the decoder_type is not 'euc' or 'hyp'.

        """
        super().__init__()
        self.manifold_in = Lorentz(k=float(args.k_in))
        # self.manifold_hidden = Lorentz(k=float(args.k_in))
        self.manifold_hidden = Lorentz(k=float(args.k_out))
        self.decoder_type = args.decoder_type

        self.manifold_out = Lorentz(k=float(args.k_out))
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.trans_conv = TransConv(self.manifold_in, self.manifold_hidden, self.manifold_out, in_channels, hidden_channels, trans_num_layers, trans_num_heads, trans_dropout, trans_use_bn, trans_use_residual, trans_use_weight, trans_use_act, args)

        self.aggregate = aggregate

        if self.decoder_type == 'euc':
            self.decode_trans = nn.Linear(self.hidden_channels, self.out_channels)
            self.decode_graph = nn.Linear(self.hidden_channels, self.out_channels)
        elif self.decoder_type == 'hyp':
            self.decode_graph = HypLinear(self.manifold_out, self.hidden_channels, self.hidden_channels)
            self.decode_trans = HypCLS(self.manifold_out, self.hidden_channels, self.out_channels)
        else:
            raise NotImplementedError

    def forward(self, x):
        x1 = self.trans_conv(x)
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

    def reset_parameters(self):
        if self.use_graph:
            self.graph_conv.reset_parameters()