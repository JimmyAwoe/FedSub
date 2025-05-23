export CUDA_VISIBLE_DEVICES=0,1,2,3
opt=("subscafsgd")
#opt=("sgd" "subscafsgd")
for opt in "${opt[@]}"
do
torchrun --nproc-per-node 2 --master-port 25902 SubspaceScaffold.py \
    --comp_dim 128 \
    --model_config configs/llama_60m.json \
    --optimizer "${opt}" \
    --max_length 512 \
    --batch_size 64\
    --total_batch_size 64 \
    --warmup 1000\
    --tau 16 \
    --lr 1e-3 \
    --use_wandb \
    --per_layer_weight_update \
    --momentum 0.9 \
    --wandb_run_name "tau2" \
    --gene_method cd
done