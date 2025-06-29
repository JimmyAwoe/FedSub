export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
opt=("subscafsgd")
for opt in "${opt[@]}"
do
torchrun --nproc-per-node 2 --master-port 25900 resnet.py \
    --comp_dim 3 \
    --arch resnet110 \
    --optimizer "${opt}" \
    --batch_size 64 \
    --tau 10 \
    --epochs 5 \
    --lr 0.1 \
    --warmup 100 \
    --update_cp_freq 50 \
    --use_log \
    --data_hete \
    --print-freq 10 \
    --gene_method cd \
    > "./logs/${opt}_resnet_train.log" 2>&1
done

    #--adaptive_cp_rate 1 \

