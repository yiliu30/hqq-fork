import logging
import gc
import psutil

from hqq.core.common.config import opition

_cuda_accelerator = None

if opition.use_cuda:
    from hqq.core.common.cuda_accelerator import CUDA_Accelerator

    _cuda_accelerator = CUDA_Accelerator()


def get_accelerator():
    return _cuda_accelerator


# Create a logger with a unique name (usually the module name)
logger = logging.getLogger("hqq")

torch_memory_reserved = get_accelerator().memory_reserved
torch_max_memory_reserved = get_accelerator().max_memory_reserved


def see_memory_usage(message, force=False):
    # if dist.is_initialized() and not dist.get_rank() == 0:
    #     return

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    # Print message except when distributed but not rank 0
    print(message)
    print(
        f"MA {round(get_accelerator().memory_allocated() / (1024 * 1024 * 1024),2 )} GB \
        Max_MA {round(get_accelerator().max_memory_allocated() / (1024 * 1024 * 1024),2)} GB \
        CA {round(torch_memory_reserved() / (1024 * 1024 * 1024),2)} GB \
        Max_CA {round(torch_max_memory_reserved() / (1024 * 1024 * 1024))} GB "
    )

    vm_stats = psutil.virtual_memory()
    used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
    print(f"CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%")

    # get the peak memory to report correct data, so reset the counter for the next call
    get_accelerator().reset_peak_memory_stats()
