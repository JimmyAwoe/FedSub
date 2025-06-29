export CUDA_VISIBLE_DEVICES=0,1
#optim=("subscafsgd" "sgd")
optim=("fedavgsgd" "subscafsgd")
for opt in "${optim[@]}"
do
torchrun --nproc-per-node 2 --master-port 25902 llama_finetune.py \
    --optimizer "${opt}" \
    --max_length 512 \
    --batch_size 16 \
    --total_batch_size 16 \
    --warmup 0 \
    --tau 5 \
    --lr 1e-4 \
    --constant_lr \
    --adaptive_cp_rate 0.7 \
    --comp_dim 256 \
    --use_log \
    --update_cp_freq 50 \
    --flash_attn \
    --gene_method cd \
    --epoch 1 \
    --eval_freq 1000 \
    > "./logs/${opt}_finetune.log" 2>&1 
done

torchrun --nproc-per-node 2 --master-port 25902 llama_finetune.py \
    --optimizer "${opt}" \
    --max_length 512 \
    --batch_size 16 \
    --total_batch_size 16 \
    --warmup 0 \
    --tau 5 \
    --lr 1e-4 \
    --constant_lr \
    --adaptive_cp_rate 1 \
    --comp_dim 256 \
    --use_log \
    --update_cp_freq 50 \
    --flash_attn \
    --gene_method cd \
    --epoch 1 \
    --eval_freq 1000 \
    > "./logs/subscafsgd_full_finetune.log" 2>&1 
