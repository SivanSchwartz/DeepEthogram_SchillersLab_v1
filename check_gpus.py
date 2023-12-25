import torch

def print_gpus():
    """Simple function to print available GPUs, their name, device capability, and memory.
    Note: sometimes torch.cuda.max_memory_allocated does not actually match available VRAM.
    """
    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        print('GPU %d %s: Compute Capability %d.%d, Mem:%f' %
              (i, torch.cuda.get_device_name(i), int(torch.cuda.get_device_capability(i)[0]),
               int(torch.cuda.get_device_capability(i)[1]), torch.cuda.max_memory_allocated(i)), flush=True)
        

if __name__ == '__main__':
    print_gpus()