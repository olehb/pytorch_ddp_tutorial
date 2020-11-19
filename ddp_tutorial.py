import torch
from torchvision import datasets, transforms
from torch import nn, optim

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
                                  train=True,
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
        nn.Linear(28*28, 128),
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
        for x, y in train_loader:
            x = x.view(x.shape[0], -1)
            optimizer.zero_grad()
            y_hat = model(x)
            batch_loss = loss(y_hat, y)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item() / x.shape[0]

        # calculate validation loss
        with torch.no_grad():
            val_loss = 0
            for x, y in test_loader:
                x = x.view(x.shape[0], -1)
                y_hat = model(x)
                batch_loss = loss(y_hat, y)
                val_loss += batch_loss.item() / x.shape[0]

        print(f"Epoch={i}, train_loss={epoch_loss}, val_loss={val_loss}")

    return model


if __name__ == '__main__':
    model = main()
    # serialize and publish the model etc.

