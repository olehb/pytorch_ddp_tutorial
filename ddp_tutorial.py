import argparse
from typing import Tuple
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.distributed import Backend
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


def create_data_loaders(rank: int,
                        world_size: int,
                        batch_size: int) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset_loc = './mnist_data'

    train_dataset = datasets.MNIST(dataset_loc,
                                   download=True,
                                   train=True,
                                   transform=transform)
    sampler = DistributedSampler(train_dataset,
                                 num_replicas=world_size,  # Number of GPUs
                                 rank=rank,  # GPU where process is running
                                 shuffle=True,  # Shuffling is done by Sampler
                                 seed=42)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False,  # This is mandatory to set this to False here, shuffling is done by Sampler
                              num_workers=0,  # This is important to load data in the same process, so it should be 0
                              sampler=sampler,
                              pin_memory=True)

    # This is not necessary to use distributed sampler for the test or validation sets.
    test_dataset = datasets.MNIST(dataset_loc,
                                  download=True,
                                  train=False,
                                  transform=transform)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0,
                             pin_memory=True)

    return train_loader, test_loader


def create_model():
    # create model architecture
    model = nn.Sequential(
        nn.Linear(28*28, 128),  # MNIST images are 28x28 pixels
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10, bias=False)  # 10 classes to predict
    )
    return model


def main(rank: int,
         model: nn.Module,
         train_loader: DataLoader,
         test_loader: DataLoader) -> nn.Module:
    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    # initialize optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss()

    # train the model
    for i in range(10):
        model.train()
        train_loader.sampler.set_epoch(i)

        epoch_loss = 0
        # train the model for one epoch
        pbar = tqdm(train_loader)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            x = x.view(x.shape[0], -1)
            optimizer.zero_grad()
            y_hat = model(x)
            batch_loss = loss(y_hat, y)
            batch_loss.backward()
            optimizer.step()
            batch_loss_scalar = batch_loss.item()
            epoch_loss += batch_loss_scalar / x.shape[0]
            pbar.set_description(f'training batch_loss={batch_loss_scalar:.4f}')

        # calculate validation loss
        with torch.no_grad():
            model.eval()
            val_loss = 0
            pbar = tqdm(test_loader)
            for x, y in pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                x = x.view(x.shape[0], -1)
                y_hat = model(x)
                batch_loss = loss(y_hat, y)
                batch_loss_scalar = batch_loss.item()

                val_loss += batch_loss_scalar / x.shape[0]
                pbar.set_description(f'validation batch_loss={batch_loss_scalar:.4f}')

        print(f"Epoch={i}, train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}")

    return model.module


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    rank = args.local_rank
    world_size = torch.cuda.device_count()

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend=Backend.NCCL,
                                         init_method='env://')

    train_loader, test_loader = create_data_loaders(rank, world_size, 128)
    model = main(rank=rank,
                 model=create_model(),
                 train_loader=train_loader,
                 test_loader=test_loader)

    if rank == 0:
        torch.save(model.state_dict(), 'model.pt')
