from typing import Tuple

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from tqdm import tqdm

DISABLE_TQDM = True

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
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
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

def main():
    # download train dataset
    train_dataset = datasets.MNIST('./mnist_data',
                                   download=True,
                                   train=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    # download test dataset
    test_dataset = datasets.MNIST('./mnist_data',
                                  download=True,
                                  train=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))
    # create train data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=128,
                                               shuffle=True,
                                               num_workers=4)
    # create test data loader
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=128,
                                              shuffle=True,
                                              num_workers=4)
    # create model architecture
    model = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10, bias=False)
    )

    # initialize optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss()

    # train the model
    for i in range(10):
        epoch_loss = 0
        # train the model for one epoch
        pbar = tqdm(train_loader, disable=DISABLE_TQDM)
        for x, y in pbar:
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
            val_loss = 0
            pbar = tqdm(test_loader, disable=DISABLE_TQDM)
            for x, y in pbar:
                x = x.view(x.shape[0], -1)
                y_hat = model(x)
                batch_loss = loss(y_hat, y)

                batch_loss_scalar = batch_loss.item()
                val_loss += batch_loss_scalar / x.shape[0]
                pbar.set_description(f'validation batch_loss={batch_loss_scalar:.4f}')

        print(f"Epoch={i}, train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}")

    return model


if __name__ == '__main__':
    model = main()
    # serialize and publish the model etc.
