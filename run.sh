export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc-per-node 2 --master-port 25902 SubspaceScaffold.py \
    --comp_dim 256 \
    --model_config configs/llama_60m.json \
    --optimizer subscaf \
    --batch_size 16 \
    --total_batch_size 64 \
    --warmup 1000\
    --tau 16 \
    --use_wandb \
    --lr 1e-3 \