export CUDA_VISIBLE_DEVICES=0,1,2,3
opt=("subscafsgd" "fedavgsgd")
sl=1024
bs=64
for opt in "${opt[@]}"
do
torchrun --nproc-per-node 2 --master-port 25900 llama_pretrain.py \
    --comp_dim 256 \
    --model_config configs/llama_60m.json \
    --optimizer "${opt}" \
    --max_length $sl \
    --batch_size $bs \
    --total_batch_size $bs \
    --warmup 1000 \
    --tau 10 \
    --lr 1e-3 \
    --momentum 0 \
    --constant_lr \
    --dampeniog 0 \
    --num_training_steps 3 \
    --update_cp_freq 50 \
    --mixed_precision bf16 \
    --use_log \
    --ckpt \
    --change_cd 3000 \
    --wandb_run_name "real_lazy_update" \
    --gene_method cd \
    > "./logs/${opt}_pretrain.log" 2>&1
done

torchrun --nproc-per-node 2 --master-port 25900 llama_pretrain.py \
    --adaptive_cp_rate 1 \
    --model_config configs/llama_60m.json \
    --optimizer subscafsgd \
    --max_length $sl \
    --batch_size $bs \
    --total_batch_size $bs \
    --warmup 1000 \
    --tau 10 \
    --lr 1e-3 \
    --momentum 0 \
    --constant_lr \
    --dampeniog 0 \
    --num_training_steps 3 \
    --update_cp_freq 50 \
    --mixed_precision bf16 \
    --use_log \
    --ckpt \
    --change_cd 3000 \
    --wandb_run_name "real_lazy_update" \
    --gene_method idx \
    > "./logs/subscafsgd_full_pretrain.log" 2>&1


torchrun --nproc-per-node 2 --master-port 25900 llama_pretrain.py \
    --comp_dim 128 \
    --model_config configs/llama_60m.json \
    --optimizer subscafsgd \
    --max_length $sl \
    --batch_size $bs \
    --total_batch_size $bs \
    --warmup 1000 \
    --tau 10 \
    --lr 1e-3 \
    --momentum 0 \
    --constant_lr \
    --dampeniog 0 \
    --num_training_steps 3 \
    --update_cp_freq 50 \
    --mixed_precision bf16 \
    --use_log \
    --ckpt \
    --change_cd 3000 \
    --wandb_run_name "real_lazy_update" \
    --gene_method cd \
    > "./logs/cd128_pretrain.log" 2>&1


torchrun --nproc-per-node 2 --master-port 25900 llama_pretrain.py \
    --comp_dim 64 \
    --model_config configs/llama_60m.json \
    --optimizer subscafsgd \
    --max_length $sl \
    --batch_size $bs \
    --total_batch_size $bs \
    --warmup 1000 \
    --tau 10 \
    --lr 1e-3 \
    --momentum 0 \
    --constant_lr \
    --dampeniog 0 \
    --num_training_steps 3 \
    --update_cp_freq 50 \
    --mixed_precision bf16 \
    --use_log \
    --ckpt \
    --change_cd 3000 \
    --wandb_run_name "real_lazy_update" \
    --gene_method cd \
    > "./logs/cd64_pretrain.log" 2>&1