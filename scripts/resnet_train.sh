export CUDA_VISIBLE_DEVICES=0,1
opt=("subscafsgd" "fedavgsgd")
for opt in "${opt[@]}"
do
torchrun --nproc-per-node 2 --master-port 25900 resnet.py \
    --comp_dim 3 \
    --arch resnet110 \
    --optimizer "${opt}" \
    --batch_size 64 \
    --tau 5 \
    --epochs 20 \
    --lr 0.1 \
    --warmup 100 \
    --update_cp_freq 50 \
    --use_log \
    --data_hete \
    --print-freq 1 \
    --gene_method cd \
    > "./logs/${opt}_resnet_train.log" 2>&1
done

torchrun --nproc-per-node 2 --master-port 25900 resnet.py \
    --comp_dim 7 \
    --arch resnet110 \
    --optimizer subscafsgd \
    --batch_size 64 \
    --tau 5 \
    --epochs 20 \
    --lr 0.1 \
    --warmup 100 \
    --update_cp_freq 50 \
    --use_log \
    --data_hete \
    --print-freq 1 \
    --gene_method cd \
    > "./logs/subscafsgd_full_resnet_train.log" 2>&1

