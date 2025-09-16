export CUDA_VISIBLE_DEVICES=0,1
opts=("subscafsgd" "subscafsgd" "subscafsgd" "fedavgsgd" "subscafsgd")
 # 1000000 means do NOT compression
comp_dims=(1000000 1000000 1000000 100000 100000)
adaptive_cp_rates=(0.7 0.7 0.7 1 1)
gene_methods=("cd" "cd" "cd" "idx" "idx")
sl=512
bs=16
for opt in "${optim[@]}"
do
    opt=${opts[$i]}
    comp_dim=${comp_dims[$i]}
    adaptive_cp_rate=${adaptive_cp_rates[$i]}
    gene_method=${gene_methods[$i]}
    torchrun --nproc-per-node 2 --master-port 25902 llama_finetune.py \
        --optimizer "${opt}" \
        --max_length $sl \
        --batch_size $bs \
        --total_batch_size 16 \
        --warmup 0 \
        --tau 5 \
        --lr 1e-4 \
        --constant_lr \
        --adaptive_cp_rate $adaptive_cp_rate \
        --comp_dim $comp_dim \
        --use_log \
        --update_cp_freq 50 \
        --flash_attn \
        --gene_method cd \
        --epoch 1 \
        --eval_freq 1000
done
