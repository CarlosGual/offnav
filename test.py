import os

import torch
from torch import nn
import multiprocessing as mp

torch.distributed.init_process_group(backend="nccl")
rank = int(os.environ["LOCAL_RANK"])
device = f"cuda:{rank}"
torch.cuda.set_device(device)


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


if __name__ == "__main__":
    model = ToyModel()
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], broadcast_buffers=False
    )
    print('Model created on the rank: {}'.format(rank))
    print('The device is: {}'.format(device))
    print('the number of cpus (cores) are: {}'.format(mp.cpu_count()))
    print('The number of gpus are: {}'.format(torch.cuda.device_count()))
    print('the number of available threads are: {}'.format(torch.get_num_threads()))