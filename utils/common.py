import torch.distributed as dist
from loguru import logger
import torch

def init_process_group():
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )

def log(message):
    if dist.get_rank() == 0:
        logger.info(message)

def set_seed(seed):
    # Set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
