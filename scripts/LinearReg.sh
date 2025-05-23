#!/bin/sh

python3 LinearRegExp/linearplot.py \
    --lr 0.0005 \
    --inner_iter_num 10 \
    --out_iter_num 5000 \
    --grad_noise 0 \
    --worker_num 20 \
    --cp_rank 5 \
    --cp_gene_method rd \
    --epochs 1 \
    --dim 10 \
    --total_sample 10000 \
    --grad_down_lr \
    --hete \
    --grad_clip 30 \
    #--grad_divide 100 \
    #--lbd_clip 30 \
    #--dual_re_proj \
