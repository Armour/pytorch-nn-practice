
# coding: utf-8

# In[26]:


from __future__ import print_function

import os
import math
import argparse

import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
import torch.utils as utils

from model.vgg import VGG

from torch.autograd import Variable
from torchvision import models, datasets, transforms


# In[27]:


# Args
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--learning-rate', type=float, default=0.1, 
                    help='initial learning rate (default: 0.1)')
parser.add_argument('--train-batch-size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=100,
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train (default: 500)')
parser.add_argument('--lr-decay-interval', type=int, default=100,
                    help='number of epochs to decay the learning rate (default: 100)')
parser.add_argument('--num-workers', type=int, default=4,
                    help='number of workers (default: 4)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='how many batches to wait before logging training status (default: 10)')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume from checkpoint')
args = parser.parse_args([])


# In[28]:


# Init variables
print('==> Init variables..')
use_cuda = cuda.is_available()
best_accuracy = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

torch.manual_seed(args.seed)
if use_cuda:
    cuda.manual_seed(args.seed)


# In[29]:


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    mean = torch.zeros(3)
    std = torch.zeros(3)
    dataloader = utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


# In[30]:


# Data
print('==> Download data..')
dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transforms.ToTensor())

print('==> Computing mean and std..')
data_mean, data_std = get_mean_and_std(dataset)

print(data_mean)
print(data_std)

transform_train = transforms.Compose([
    transforms.Scale(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(data_mean, data_std),
])
transform_test = transforms.Compose([
    transforms.Scale(224),
    transforms.ToTensor(),
    transforms.Normalize(data_mean, data_std),
])

print('==> Init dataloader..')
trainset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
trainloader = utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)

testset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
testloader = utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)


# In[31]:


# Model
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_accuracy = checkpoint['accuracy']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    net = VGG('vgg16', num_classes=10)

if use_cuda:
    net = net.cuda()


# In[32]:


# Loss function and Optimizer
criterion = nn.CrossEntropyLoss()
if use_cuda:
    criterion = criterion.cuda()

optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=5e-4)


# In[33]:


def train(epoch):
    """Traning epoch."""
    print('==> Training Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        print('Training loss: %f    Correct number: %f' % train_loss, correct)


# In[34]:


def test(epoch):
    """Testing epoch."""
    global best_accuracy
    print('==> Testing Epoch: %d' % epoch)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        print('Testing loss: %f    Correct number: %f' % test_loss, correct)

    # Save checkpoint.
    accuracy = 100.*correct/total
    if accuracy > best_accuracy:
        print('==> Saving checkpoint..')
        state = {
            'net': net.module if use_cuda else net,
            'accuracy': accuracy,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_accuracy = accuracy
        


# In[42]:


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial learning rate decayed by 10 every args.lr_decay_interval epochs."""
    learning_rate = args.learning_rate * (0.1 ** (epoch // args.lr_decay_interval))
    print('==> Change learning rate: %f' % learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


# In[43]:


for epoch in range(start_epoch, start_epoch + args.epochs):
    adjust_learning_rate(optimizer, epoch)
    train(epoch)
    test(epoch)

