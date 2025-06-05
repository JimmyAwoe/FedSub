export CUDA_VISIBLE_DEVICES=0,1,2,3
opt=("sgd")
#opt=("sgd" "subscafsgd")
for opt in "${opt[@]}"
do
nohup torchrun --nproc-per-node 2 --master-port 25902 llama_pretrain.py \
    --comp_dim 256 \
    --model_config configs/llama_60m.json \
    --optimizer "${opt}" \
    --max_length 256 \
    --batch_size 256 \
    --total_batch_size 256 \
    --warmup 1000 \
    --tau 3 \
    --lr 1e-2 \
    --momentum 0.9 \
    --dampening 0.9 \
    --constant_lr \
    --per_layer_weight_update \
    --num_training_steps 10000 \
    --update_cp_freq 4 \
    --use_log \
    --wandb_run_name "real_lazy_update" \
    --gene_method cd \
    > "./logs/${opt}_pretrain.log" 2>&1
done