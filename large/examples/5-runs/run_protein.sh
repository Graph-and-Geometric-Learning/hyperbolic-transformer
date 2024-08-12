#!/bin/bash

hidden_channels=256
lr=0.001
graph_weight=0.2
k_in=0.5
k_out=1.0
gnn_dropout=0
trans_dropout=0.0
weight_decay=0.0
batch_size=10000
power_k=1.0

python main-batch.py \
  --method hypformer \
  --dataset ogbn-proteins \
  --metric rocauc \
  --lr $lr \
  --hidden_channels $hidden_channels \
  --gnn_num_layers 2 \
  --gnn_dropout $gnn_dropout \
  --gnn_use_residual 1 \
  --gnn_use_weight 1 \
  --gnn_use_bn 1 \
  --gnn_use_act 1 \
  --trans_num_layers 1 \
  --trans_num_heads 1 \
  --trans_dropout $trans_dropout \
  --weight_decay $weight_decay \
  --trans_use_residual 1 \
  --trans_use_weight 1 \
  --graph_weight $graph_weight \
  --batch_size $batch_size \
  --seed 123 \
  --runs 5 \
  --epochs 500 \
  --eval_step 5 \
  --device 0 \
  --power_k $power_k \
  --data_dir $hypformer_data_dir \
  --decoder_type euc \
  --attention_type linear_focused