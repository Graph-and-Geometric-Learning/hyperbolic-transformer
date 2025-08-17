#!/bin/bash

# test accuracy on L40s: 0.8773

python main.py \
    --dataset mini \
    --lr 0.001 \
    --hidden_channels 256 \
    --epoch 1000 \
    --patience 200 \
    --method hypformer \
    --use_graph 0 \
    --trans_num_layers 2 \
    --trans_num_heads 2 \
    --trans_heads_concat 0 \
    --trans_use_weight 1 \
    --trans_use_residual 1 \
    --trans_use_bn 1 \
    --trans_use_act 1 \
    --trans_dropout 0.4 \
    --device 0 \
    --decoder_type euc \
    --attention_type linear_focused \
    --power_k 2.0 \
    --rand_split 1 \
    --k_in 2.0 \
    --k_out 3.0 \
    --runs 5 \
    --data_dir ../../data/ \
    --seed 42 \
    --train_prop 0.5 \
    --valid_prop 0.25 \
    --protocol semi \
    --no_feat_norm 1 \
    --sub_dataset gcn_data \
    --add_positional_encoding 1 \
    --optimizer_type adam \
    --weight_decay 0.005 \
    --use_wandb 0
