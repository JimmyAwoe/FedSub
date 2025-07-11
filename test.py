import torch
import torch.distributed as dist


# 初始化分布式环境
dist.init_process_group(backend="nccl")

# 检查当前GPU与其他GPU的连接方式
local_rank = dist.get_rank()
device = torch.device(f"cuda:{local_rank}")

# 使用NCCL的检查函数
if hasattr(torch.cuda.nccl, 'get_peer_access'):
    for peer_rank in range(dist.get_world_size()):
        if peer_rank != local_rank:
            # 检查是否可以直接访问对端GPU内存
            can_access = torch.cuda.nccl.get_peer_access(local_rank, peer_rank)
            print(f"Rank {local_rank} 能否直接访问 Rank {peer_rank}: {can_access}")
            # 注：若返回True，通常表示使用NVLink连接

print("finish")