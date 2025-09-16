export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
opts=("subscafsgd" "fedavgsgd" "subscafsgd")
comp_dims=(3 7 7)
log_suffixes=("" "" "_full")
gene_methods=("cd" "cd" "idx")
for ((i=0; i<${#opts[@]}; i++))
do
    opt=${opts[$i]}
    comp_dim=${comp_dims[$i]}
    log_suffix=${log_suffixes[$i]}
    gene_method=${gene_methods[$i]}
    torchrun --nproc-per-node 8 --master-port 25900 resnet.py \
        --comp_dim ${comp_dim} \
        --arch resnet110 \
        --optimizer ${opt} \
        --batch_size 32 \
        --tau 10 \
        --epochs 41 \
        --evaluate \
        --constant_lr \
        --lr 0.1 \
        --warmup 100 \
        --update_cp_freq 50 \
        --use_log \
        --data_hete \
        --print-freq 1 \
        --gene_method ${gene_method}
done

