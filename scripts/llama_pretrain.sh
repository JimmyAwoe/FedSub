export CUDA_VISIBLE_DEVICES=0,1,2,3
opt=("subscafsgd" "sgd")
for opt in "${opt[@]}"
do
nohup torchrun --nproc-per-node 2 --master-port 25900 llama_pretrain.py \
    --comp_dim 64 \
    --model_config configs/llama_350m.json \
    --optimizer "${opt}" \
    --max_length 1024 \
    --batch_size 16 \
    --total_batch_size 16 \
    --warmup 1000 \
    --tau 5 \
    --lr 1e-3 \
    --momentum 0 \
    --constant_lr \
    --mixed_precision bf16 \
    --dampeniog 0 \
    --num_training_steps 10 \
    --update_cp_freq 2 \
    --use_log \
    --wandb_run_name "real_lazy_update" \
    --gene_method cd \
    > "./logs/${opt}_pretrain.log" 2>&1
done

    #--adaptive_cp_rate 0.5 \