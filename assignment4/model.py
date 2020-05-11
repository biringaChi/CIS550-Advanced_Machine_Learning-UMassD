__author__ = "biringaChi"
__email__ = "biringachidera@gmail.com"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import time
import numpy as np
from tqdm import tqdm
import os
import pickle

os.environ['KMP_DUPLICATE_LIB_OK']='True'
seed = 60
np.random.seed(seed)
torch.manual_seed(seed)
NUM_TRAIN = 49_000

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


def data_prep(transform):
    cifar10_train = dataset.CIFAR10('./data', train=True, download=True,
                                 transform=transform)
    loader_train = DataLoader(cifar10_train, batch_size=64,
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
    cifar10_val = dataset.CIFAR10('./data', train=True, download=True,
                               transform=transform)
    loader_val = DataLoader(cifar10_val, batch_size=64,
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50_000)))
    cifar10_test = dataset.CIFAR10('./data', train=False, download=True,
                                transform=transform)
    loader_test = DataLoader(cifar10_test, batch_size=64)

    return loader_train, loader_val, loader_test

loader_train, loader_val, loader_test = data_prep(transform)


class Net(nn.Module):
    """CNN for image classifier"""
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)
        self.dropout = nn.Dropout(p=0.25)

        self.model = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            nn.Flatten(),
            self.dropout,
            self.fc1,
            self.dropout,
            self.fc2
        )

    def forward(self, x):
        x = self.model(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
epochs = 10
validation_loss_min = np.Inf

def accuracy():
    net.eval()
    correct = 0
    total = 0
    for data in loader_test:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        acc = 100 * (correct/total)
    return acc

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

epoch_train_accuracies = []
epoch_train_losses = []
epoch_val_accuracies = []
epoch_val_losses = []


start_time = time.time()
for epoch in range(epochs):
    training_loss = 0.0
    validation_loss = 0.0

    net.train()
    for i, data in tqdm(enumerate(loader_train, 0)):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

        if i % 100 == 0:
            print(f"Training Loss = {loss.item()}")
            train_losses.append(loss.item())
            print(f"Training Accuracy = {accuracy()}")
            train_accuracies.append(accuracy())

    epoch_train_losses.append(loss.item())
    epoch_train_accuracies.append(accuracy())

    net.eval()
    for i, data in tqdm(enumerate(loader_val, 0)):
        inputs, labels = data
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        validation_loss += loss.item()

        if i % 2.5 == 0:
            print(f"Validation Loss = {loss.item()}")
            val_losses.append(loss.item())
            print(f"Validation Accuracy = {accuracy()}")
            val_accuracies.append(accuracy())

    epoch_val_losses.append(loss.item())
    epoch_val_accuracies.append(accuracy())

    print(f"Average training loss: {training_loss} \nAverage validation loss: {validation_loss}")

    print(f"Epoch: {epoch}")

print(f"Finshed training. \nTime to train: {time.time() - start_time}")

directory = "./model.pth"
torch.save(net.state_dict(), directory)

# iter metric
train_loss = open("train_losses.pickle","wb")
pickle.dump(train_losses, train_loss)
train_loss.close()

val_loss = open("val_losses.pickle","wb")
pickle.dump(val_losses, val_loss)
val_loss.close()

train_accuracy = open("train_accuracies.pickle","wb")
pickle.dump(train_accuracies, train_accuracy)
train_accuracy.close()

val_accuracy = open("val_accuracies.pickle","wb")
pickle.dump(val_accuracies, val_accuracy)
val_accuracy.close()

# Epoch metric
epoch_train_acc = open("epoch_train_accuracies.pickle","wb")
pickle.dump(epoch_train_accuracies, epoch_train_acc)
epoch_train_acc.close()

epoch_train_loss = open("epoch_train_losses.pickle","wb")
pickle.dump(epoch_train_losses, epoch_train_loss)
epoch_train_loss.close()

epoch_val_loss = open("epoch_val_losses.pickle","wb")
pickle.dump(epoch_val_losses, epoch_val_loss)
epoch_val_loss.close()

epoch_val_acc = open("epoch_val_accuracies.pickle","wb")
pickle.dump(epoch_val_accuracies, epoch_val_acc)
epoch_val_acc.close()


# best testing
directory = "./model.pth"
net.load_state_dict(torch.load(directory))

def test_accuracy():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader_test:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc = 100 * (correct/total)
        return acc

print(f"Current best test: {test_accuracy()}")
