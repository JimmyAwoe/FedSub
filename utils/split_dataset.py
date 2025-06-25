import torch
from torch.utils.data import Subset, DataLoader
from torch.utils.data.distributed import DistributedSampler
def split_dataset_by_class(dataset, node_rank, world_size, batch_size):
    targets = torch.tensor(dataset.targets)
    classes_per_node = len(dataset.classes) // world_size 
    assigned_classes = range(node_rank * classes_per_node, 
                           (node_rank + 1) * classes_per_node)
    
    # 获取属于当前节点类别的样本索引
    indices = [i for i, target in enumerate(targets) if target in assigned_classes]
    
    train_subset = Subset(dataset, indices)

    train_sampler = DistributedSampler(
        train_subset, 
        num_replicas=1, 
        rank=0,
        shuffle=True  # 仍可在本地类别内打乱
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4
    )
    return train_loader