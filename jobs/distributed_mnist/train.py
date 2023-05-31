import os

import torch
import torch.nn.functional as F
import torchvision
import wandb
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import MNIST

BACKEND = os.environ.get("PL_TORCH_DISTRIBUTED_BACKEND", "nccl")
CUDA = torch.cuda.is_available()

DEFAULT_CONFIG = {
    "epochs": 0,
    "batch_size": 32,
    "lr": 1e-3,
}


def get_mnist_dataset():
    """Return pytorch MNIST dataset."""
    return MNIST(
        root="data",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )


def mnist_train():
    """Run training on the mnist dataset."""
    wandb.init(config=DEFAULT_CONFIG)
    dataset = get_mnist_dataset()
    model = torch.nn.Linear(28 * 28, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config.lr)

    if CUDA:
        model = model.cuda()

    model = DDP(model)

    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=wandb.config.batch_size, sampler=sampler)

    for epoch in range(wandb.config.epochs):
        sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(loader):
            if CUDA:
                data = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            optimizer.zero_grad()
            output = model(data.view(data.shape[0], -1))
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item()}")


if __name__ == "__main__":
    init_process_group(backend=BACKEND)
    mnist_train()
    destroy_process_group()
