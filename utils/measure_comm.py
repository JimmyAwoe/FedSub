import torch
import torch.distributed as dist

def measure_all_reduce(tensor, op):
    """测量all_reduce操作的耗时"""
    torch.cuda.synchronize()
    dist.barrier()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
        
    dist.all_reduce(tensor, op=op)
        
    end_event.record()
    torch.cuda.synchronize()
        
    return start_event.elapsed_time(end_event)
        #"bandwidth_GBps": (tensor.numel() * 4 * 2) / (avg_time * 1e6)  # 往返通信量/时间

def measure_broadcast(tensor, src):
    torch.cuda.synchronize()
    dist.barrier()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
        
    dist.broadcast(tensor, src=src)
        
    end_event.record()
    torch.cuda.synchronize()
        
    return start_event.elapsed_time(end_event)


    
