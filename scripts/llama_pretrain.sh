export CUDA_VISIBLE_DEVICES=0,1,2,3
opt=("subscafsgd")
#opt=("sgd" "subscafsgd")
for opt in "${opt[@]}"
do
nohup torchrun --nproc-per-node 2 --master-port 25902 SubspaceScaffold.py \
    --comp_dim 256 \
    --model_config configs/llama_60m.json \
    --optimizer "${opt}" \
    --max_length 512 \
    --batch_size 64\
    --total_batch_size 64 \
    --warmup 1000\
    --tau 5 \
    --lr 1e-3 \
    --use_wandb \
    --per_layer_weight_update \
    --nesterov \
    --momentum 0.9 \
    --weight_decay 0.01 \
    --update_cp_freq 40 \
    --wandb_run_name "real_lazy_update" \
    --gene_method cd
    > "./logs/llama_pretrain.log" 2>&1 &
done