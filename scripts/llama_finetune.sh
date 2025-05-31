export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc-per-node 2 --master-port 25902 llama_finetune.py \
    --comp_dim 256 \
    --optimizer "sgd" \
    --max_length 256 \
    --batch_size 8 \
    --total_batch_size 8 \
    --warmup 0 \
    --tau 10 \
    --lr 1e-4 \
    --per_layer_weight_update \
    --adaptive_cp_rate 0.5 \
    --nesterov \
    --momentum 0.9 \
    --weight_decay 0 \
    --update_cp_freq 4 \
    --wandb_run_name "vanilla finetune" \
    --gene_method cd \
    > "./logs/llama_finetune_sgd.log" 2>&1
    #> "./logs/llama_finetune.log" 2>&1 
