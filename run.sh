export CUDA_VISIBLE_DEVICES=0,1,2,3
opt=("subscafsgd")
#opt=("sgd" "subscafsgd")
for opt in "${opt[@]}"
do
torchrun --nproc-per-node 2 --master-port 25902 SubspaceScaffold.py \
    --comp_dim 128 \
    --model_config configs/llama_60m.json \
    --optimizer "${opt}" \
    --batch_size 32 \
    --total_batch_size 32 \
    --warmup 1000\
    --tau 16 \
    --lr 1e-3 \
    --use_wandb \
    --per_layer_weight_update \
    --momentum 0.9 \
    --wandb_run_name "coff-${opt}" \
    --gene_method cd
done