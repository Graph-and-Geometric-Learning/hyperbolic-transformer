#!/bin/bash
graph_weight=0.2
k_in=0.5
k_out=1.0
power_k=2.0
lr=0.005

python main-batch.py \
  --method hypformer \
  --dataset amazon2m \
  --metric acc \
  --lr $lr \
  --hidden_channels 256 \
  --gnn_num_layers 3 \
  --gnn_dropout 0.0 \
  --weight_decay 0. \
  --gnn_use_residual 1 \
  --gnn_use_weight 1 \
  --gnn_use_bn 1 \
  --gnn_use_init 1 \
  --gnn_use_act 1 \
  --trans_num_layers 1 \
  --trans_dropout 0. \
  --trans_use_residual 1 \
  --trans_use_weight 1 \
  --trans_use_bn 1 \
  --use_graph 1 \
  --graph_weight $graph_weight \
  --batch_size 100000 \
  --seed 123 \
  --runs 5 \
  --epochs 200 \
  --eval_step 1 \
  --device 0 \
  --k_in $k_in \
  --k_out $k_out \
  --power_k $power_k \
  --attention_type linear_focused

