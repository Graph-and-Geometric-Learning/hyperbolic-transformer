#!/bin/bash

# Highest Test: 74.40 ± 0.26 Final Test: 73.32 ± 0.50
# activate conda env before running

python main.py \
    --dataset citeseer  \
    --method hypformer \
    --lr 0.005  \
    --hidden_channels 256  \
    --use_graph 1  \
    --weight_decay 0.005  \
    --gnn_num_layers 5 \
    --graph_weight 0.4 \
    --gnn_dropout 0.5 \
    --gnn_use_weight 0 \
    --gnn_use_bn 0  \
    --gnn_use_residual 1  \
    --gnn_use_init 0  \
    --gnn_use_act 1  \
    --trans_num_layers 1  \
    --trans_dropout 0.5  \
    --trans_use_residual 1  \
    --trans_use_weight 1  \
    --trans_num_heads 1  \
    --trans_use_bn 0  \
    --trans_use_act 0  \
    --rand_split_class 1  \
    --valid_num 500  \
    --test_num 1000  \
    --no_feat_norm 1  \
    --add_positional_encoding 1  \
    --epochs 500 \
    --seed 123  \
    --device 0  \
    --runs 1  \
    --power_k 3.0  \
    --k_in 1.0 \
    --k_out 1.0 \
    --attention_type linear_focused \
    --decoder_type euc \
    --save_result 0
