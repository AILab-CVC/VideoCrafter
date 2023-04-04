import torch
import torch.distributed as dist

def setup_dist(local_rank):
    if dist.is_initialized():
        return
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )

def gather_data(data, return_np=True):
    ''' gather data from multiple processes to one list '''
    data_list = [torch.zeros_like(data) for _ in range(dist.get_world_size())]
    dist.all_gather(data_list, data)  # gather not supported with NCCL
    if return_np:
        data_list = [data.cpu().numpy() for data in data_list]
    return data_list
