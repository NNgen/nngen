from __future__ import absolute_import
from __future__ import print_function

import time
import copy
import datetime

import torch
import torchvision
import torchvision.transforms as transforms


# --------------------------------------
# Dataset preparation
# --------------------------------------

batch_size = 128
num_epochs = 100

# train
train_transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = torchvision.datasets.CIFAR10(root='../cifar10data', train=True,
                                         download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=2)

# validation
validation_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
validation_set = torchvision.datasets.CIFAR10(root='../cifar10data', train=True,
                                              download=True, transform=validation_transform)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

# test
test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_set = torchvision.datasets.CIFAR10(root='../cifar10data', train=False,
                                        download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# --------------------------------------
# Model definition and customization
# --------------------------------------

import torch.nn as nn
import torch.nn.functional as F

net = torchvision.models.vgg11_bn()

#net.avgpool = nn.Identity()

net.classifier[0] = nn.Linear(512, 4096)
net.classifier[6] = nn.Linear(4096, 10)

# net.classifier = nn.Sequential(
#    nn.Linear(in_features=512, out_features=1024, bias=True),
#    nn.ReLU(inplace=True),
#    nn.Dropout(p=0.5),
#    nn.Linear(in_features=1024, out_features=1024, bias=True),
#    nn.ReLU(inplace=True),
#    nn.Dropout(p=0.5),
#    nn.Linear(in_features=1024, out_features=10, bias=True),
# )

print(net)


# --------------------------------------
# Optimizer
# --------------------------------------

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)


# ---------------------------------------
# Train
# ---------------------------------------

# train method
def train(model, train_dataloader, validation_dataloader,
          criterion, optimizer, num_epochs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    start_time = time.time()
    train_accuracy_history = []
    validation_accuracy_history = []

    best_model_params = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        print('# epoch {} / {}'.format(epoch, num_epochs - 1))

        # train
        epoch_loss, epoch_accuracy = run_epoch(device, model, train_dataloader,
                                               criterion, optimizer, is_train=True)
        train_accuracy_history.append(epoch_accuracy)
        print('[train] loss: {:.4f} accuracy: {:.4f}'.format(epoch_loss, epoch_accuracy))

        # validation
        epoch_loss, epoch_accuracy = run_epoch(device, model, validation_dataloader,
                                               criterion, optimizer, is_train=False)
        validation_accuracy_history.append(epoch_accuracy)
        print('[valid] loss: {:.4f} accuracy: {:.4f}'.format(epoch_loss, epoch_accuracy))

        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_model_params = copy.deepcopy(model.state_dict())
            print('[valid] best accuracy: {:.4f}'.format(epoch_accuracy))

    time_elapsed = time.time() - start_time
    print('# training complete in {}m {}s'.format(int(time_elapsed // 60),
                                                  int(time_elapsed % 60)))
    print('# best validation accuracy: {:4f}'.format(best_accuracy))

    model.load_state_dict(best_model_params)
    return model, train_accuracy_history, validation_accuracy_history


def run_epoch(device, model, loader, criterion, optimizer, is_train=True):

    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

        if criterion is not None:
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

        if is_train:
            loss.backward()
            optimizer.step()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_accuracy = running_corrects.double() / len(loader.dataset)

    return epoch_loss, epoch_accuracy


# call train
trained_net, train_history, validation_history = train(net, train_loader, validation_loader,
                                                       criterion, optimizer, num_epochs)

# ---------------------------------------
# Test
# ---------------------------------------


# test method
def test(model, test_dataloader):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    start_time = time.time()

    # test
    loss, accuracy = run_epoch(device, model, test_dataloader,
                               None, None, is_train=False)

    print('[test] accuracy: {:.4f}'.format(accuracy))

    time_elapsed = time.time() - start_time
    print('# test complete in {}m {}s'.format(int(time_elapsed // 60),
                                              int(time_elapsed % 60)))

    return accuracy


# call test
test_accuracy = test(net, test_loader)


# save trained model
date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
path = 'model-%s' % date

torch.save(trained_net.state_dict(), path)

# load trained model
# net.load_state_dict(torch.load(path))
