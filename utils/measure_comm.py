import torch
import torch.distributed as dist
from common import log

def measure_all_reduce(tensor, op):
    """test all_reduce operator time usage"""
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
    """test broadcast operator time usage"""
    torch.cuda.synchronize()
    dist.barrier()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
        
    dist.broadcast(tensor, src=src)
        
    end_event.record()
    torch.cuda.synchronize()
        
    return start_event.elapsed_time(end_event)

def mem():
    torch.cuda.empty_cache()
    log('memory allocated: ' + str((torch.cuda.memory_allocated() / (1024 ** 3))) + 'GB')
    log('memory reserved: ' + str(torch.cuda.memory_reserved() / (1024 ** 3)) + 'GB')
    log('max memory allocated: ' + str(torch.cuda.max_memory_allocated() / (1024 ** 3)) + 'GB')
    log('max memory reserved: ' + str(torch.cuda.max_memory_reserved() / (1024 ** 3)) + 'GB')


    
