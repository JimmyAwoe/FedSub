export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# subspace scaffold for California housing price prediction
torchrun --nproc-per-node 8 CF_housing_pred.py \
    --dim 8 \
    --rank 4 \
    --tau 10 \
    --epoch 200 \
    --batch_size 1024 \
    --lr 0.01 \
    --plot True

# baseline for California housing price prediction
torchrun --nproc-per-node 8 baseline_simple.py \
    --dim 8 \
    --epoch 20 \
    --lr 0.001