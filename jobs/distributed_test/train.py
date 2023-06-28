import os

import torch
import torch.nn.functional as F
import torchvision
import wandb
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import barrier
from torchvision.datasets import FakeData

CUDA = torch.cuda.is_available()
BACKEND = "nccl" if CUDA else "gloo"


print("CUDA Available: ", CUDA)
print("WORLD_SIZE: ", os.environ.get("WORLD_SIZE"))
print("RANK: ", os.environ.get("RANK"))
LOCAL_RANK = os.environ.get("LOCAL_RANK")
print("LOCAL_RANK: ", os.environ.get("LOCAL_RANK"))
print("MASTER_ADDR: ", os.environ.get("MASTER_ADDR"))
print("MASTER_PORT: ", os.environ.get("MASTER_PORT"))
DEVICE = torch.device(f"cuda:{LOCAL_RANK}" if CUDA else "cpu")
print("DEVICE: ", DEVICE)

DEFAULT_CONFIG = {
    "epochs": 1,
    "batch_size": 32,
    "lr": 1e-3,
}


def mnist_train():
    """Run training on the mnist dataset."""
    wandb.init(config=DEFAULT_CONFIG)
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = FakeData(
        size=1000, image_size=(3, 128, 128), num_classes=196, transform=transforms
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(49152, 4096),
        torch.nn.ReLU(),
        torch.nn.Linear(4096, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 196),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config.lr)

    if CUDA:
        print("Moving model to GPU:", DEVICE)
        model = model.to(DEVICE)

    model = DDP(model)

    sampler = DistributedSampler(
        dataset,
        num_replicas=int(os.environ.get("WORLD_SIZE")),
        rank=int(os.environ.get("RANK")),
    )

    print("Batch size:", wandb.config.batch_size)
    loader = DataLoader(dataset, batch_size=wandb.config.batch_size, sampler=sampler)

    barrier()

    for epoch in range(wandb.config.epochs):
        sampler.set_epoch(epoch)
        for _, (data, target) in enumerate(loader):
            if CUDA:
                data = data.to(DEVICE)
                target = target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data.view(data.shape[0], -1))
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss.item()})


if __name__ == "__main__":
    init_process_group(backend=BACKEND)
    mnist_train()
    destroy_process_group()
