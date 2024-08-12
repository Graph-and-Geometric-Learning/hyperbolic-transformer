#!/bin/bash

hidden_channel=256
lr=0.005
graph_weight=0.2
k_in=2.0
k_out=0.5
weight_decay=0.0

python main.py \
  --method hypformer \
  --dataset ogbn-arxiv \
  --metric acc \
  --lr $lr \
  --hidden_channels $hidden_channel \
  --gnn_num_layers 3 \
  --gnn_dropout 0.4 \
  --gnn_use_residual 1 \
  --gnn_use_weight 1 \
  --gnn_use_bn 1 \
  --gnn_use_act 1 \
  --trans_num_layers 1 \
  --trans_dropout 0. \
  --weight_decay $weight_decay \
  --trans_use_residual 1 \
  --trans_use_weight 1 \
  --trans_num_heads 2 \
  --use_graph 1 \
  --graph_weight $graph_weight \
  --seed 123 \
  --runs 1 \
  --save_result 0 \
  --epochs 1000 \
  --eval_step 1 \
  --device 0 \
  --k_in $k_in \
  --k_out $k_out \
  --attention_type linear_focused
