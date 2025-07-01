export CUDA_VISIBLE_DEVICES=0,1,2,3
opts=("subscafsgd" "subscafsgd" "subscafsgd" "fedavgsgd" "subscafsgd")
comp_dims=(256 128 64 4096 4096)
adaptive_cp_rates=(0 0 0 1 1)
log_suffixes=("256" "128" "64" "" "_full")
gene_methods=("cd" "cd" "cd" "idx" "idx")
sl=1024
bs=64
for ((i=0; i<${#opts[@]}; i++))
do
    opt=${opts[$i]}
    comp_dim=${comp_dims[$i]}
    adaptive_cp_rate=${adaptive_cp_rates[$i]}
    log_suffix=${log_suffixes[$i]}
    gene_method=${gene_methods[$i]}
    torchrun --nproc-per-node 2 --master-port 25900 llama_pretrain.py \
        --comp_dim ${comp_dim} \
        --adaptive_cp_rate ${adaptive_cp_rate} \
        --model_config configs/llama_60m.json \
        --optimizer "${opt}" \
        --max_length $sl \
        --batch_size $bs \
        --total_batch_size $bs \
        --warmup 1000 \
        --tau 10 \
        --lr 1e-3 \
        --momentum 0 \
        --measure_comm \
        --constant_lr \
        --dampeniog 0 \
        --num_training_steps 20 \
        --update_cp_freq 10 \
        --mixed_precision bf16 \
        --use_log \
        --ckpt \
        --change_cd 3000 \
        --wandb_run_name "real_lazy_update" \
        --gene_method "${gene_method}" \
        > "./logs/${opt}${log_suffix}_pretrain.log" 2>&1
done
