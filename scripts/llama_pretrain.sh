export CUDA_VISIBLE_DEVICES=0,1,2,3
opt=("subscafsgd")
for opt in "${opt[@]}"
do
nohup torchrun --nproc-per-node 2 --master-port 25900 llama_pretrain.py \
    --comp_dim 128 \
    --model_config configs/llama_60m.json \
    --optimizer "${opt}" \
    --max_length 1024 \
    --batch_size 64 \
    --total_batch_size 64 \
    --warmup 1000 \
    --tau 10 \
    --lr 1e-3 \
    --momentum 0 \
    --constant_lr \
    --dampeniog 0 \
    --num_training_steps 10000 \
    --update_cp_freq 50 \
    --mixed_precision bf16 \
    --use_log \
    --ckpt \
    --change_cd 3000 \
    --wandb_run_name "real_lazy_update" \
    --gene_method svd \
    > "./logs/${opt}_pretrain.log" 2>&1
done


    #--adaptive_cp_rate 1 \