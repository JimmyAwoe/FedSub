export CUDA_VISIBLE_DEVICES=0,1
#optim=("subscafsgd" "sgd")
optim=("subscafsgd")
for opt in "${optim[@]}"
do
torchrun --nproc-per-node 2 --master-port 25902 llama_finetune.py \
    --comp_dim 256 \
    --optimizer "${opt}" \
    --max_length 256 \
    --batch_size 8 \
    --total_batch_size 8 \
    --warmup 0 \
    --tau 3 \
    --lr 1e-4 \
    --constant_lr \
    --per_layer_weight_update \
    --adaptive_cp_rate 0.7 \
    --momentum 0 \
    --dampening 0 \
    --weight_decay 0 \
    --use_log \
    --update_cp_freq 8 \
    --wandb_run_name "vanilla finetune" \
    --gene_method rd \
    --epoch 1 \
    --eval_freq 1000 \
    > "./logs/subscaf_m.log" 2>&1 
    #> "./logs/llama_finetune_sgd.log" 2>&1
    #> "./logs/llama_finetune.log" 2>&1 
done