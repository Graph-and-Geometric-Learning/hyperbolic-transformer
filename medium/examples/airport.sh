#!/bin/bash

python main.py \
  --dataset airport \
  --method hypformer \
  --lr 0.005 \
  --weight_decay 1e-3 \
  --hidden_channels 256 \
  --use_graph 1 \
  --gnn_dropout 0.4 \
  --gnn_use_bn 1 \
  --gnn_num_layers 3 \
  --gnn_use_init 1 \
  --trans_num_layers 1 \
  --trans_use_residual 1 \
  --trans_use_bn 0 \
  --graph_weight 0.2 \
  --trans_dropout 0.2 \
  --device 0 \
  --runs 1 \
  --power_k 2.0 \
  --epochs 5000 \
  --decoder hyp \
  --k_in 1.0 \
  --k_out 2.0 \
  --data_dir ../data \
  --decoder_type hyp
