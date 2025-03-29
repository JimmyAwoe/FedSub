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

# test scaffold convergency with randomly generated data
# if you don't want to plot or test baseline, you can just delete the line
# for --plot or --baseline. Change --plot or --baseline to False would not work
torchrun --nproc-per-node 8 SS_converge_test.py \
    --dim 50 \
    --tau 10 \
    --lr 0.01 \
    --sample_num 50000 \
    --rank 10 \
    --iter_num 50000 \
    --plot True\
    --noise 0
