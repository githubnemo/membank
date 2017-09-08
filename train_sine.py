from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

import os

from membank import BlockLayer
from sine_data import train_dataset, valid_dataset

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--checkpoint', type=str, default='model.pkl')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


class BlockModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        layer0 = BlockLayer(num_blocks=5,
                            input_dim=1,
                            output_dim=10)
        layer1 = BlockLayer(num_blocks=1,
                            input_dim=10,
                            output_dim=1)

    def forward(self, x):
        l0 = self.layer0(x)
        l1 = self.layer1(l0)
        return l1

model = BlockModel()

if args.cuda:
    model.cuda()


mse = nn.MSELoss()
eps = 1e-08

optimizer = optim.SGD(model.parameters(), lr=args.lr)


X_train, y_train = train_dataset()
X_train, y_train = torch.Tensor(X_train), torch.LongTensor(y_train)

X_valid, y_valid = valid_dataset()
X_valid, y_valid = torch.Tensor(X_valid), torch.LongTensor(y_valid)

sine_train_loader = DataLoader(TensorDataset(X_train, y_train),
                               batch_size=args.batch_size,
                               shuffle=True)
sine_valid_loader = DataLoader(TensorDataset(X_valid, y_valid),
                               batch_size=args.batch_size,
                               shuffle=False)


def loss_func(pred, true):
    return mse(pred, true)

def train_sine(epoch):
    model.train()
    for batch_idx, (data, label) in enumerate(sine_train_loader):
        data = Variable(torch.Tensor(data))
        label = Variable(torch.LongTensor(label))
        if args.cuda:
            data = data.cuda()
            label = label.cuda()
        optimizer.zero_grad()
        y_pred = model(data)
        loss = loss_func(y_pred, label)
        loss.backward()
        optimizer.step()

        print("{epoch}: loss: {loss:.4f}".format(**{
            'epoch': epoch,
            'loss': loss.data[0],
        }))

def valid_sine(epoch):
    model.eval()
    for batch_idx, (data, label) in enumerate(sine_valid_loader):
        data = Variable(torch.Tensor(data))
        label = Variable(torch.LongTensor(label))
        if args.cuda:
            data = data.cuda()
            label = label.cuda()
        y_pred = model(data)
        loss = loss_func(y_pred, label)

        print("valid: {epoch}: loss: {loss}".format(**{
            'epoch': epoch,
            'loss': loss.data[0],
        }))


def main():
    try:
        for epoch in range(1, args.epochs + 1):
            train_sine(epoch)
            valid_sine(epoch)
    except KeyboardInterrupt:
        pass

    cpname = args.checkpoint.format('generic')
    with open(cpname,'wb') as f:
        torch.save(model, f)

if __name__ == "__main__":
    main()
