export CUDA_VISIBLE_DEVICES=0,1
opt=("subscafsgd")
do
torchrun --nproc-per-node 2 --master-port 25902 SubspaceScaffold.py \
    --comp_dim 256 \
    --model_config configs/llama_60m.json \
    --optimizer "${opt}" \
    --max_length 512 \
    --batch_size 64 \
    --total_batch_size 64 \
    --warmup 1000\
    --tau 5 \
    --lr 1e-3 \
    --use_wandb \
    --mixed_precision bf16 \
    --update_cp_freq 40 \
    --wandb_run_name "real_lazy_update" \
    --gene_method cd
done