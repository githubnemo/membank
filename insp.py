from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

from sine_data import *


def mase(y_t, y_p):
    T = len(y_t)
    pred_error = abs(y_t - y_p).sum()
    naive = abs(y_t[1:] - y_t[:-1]).sum()
    return pred_error / (T / (T-1) * naive)


def plot(model, num_curves=-1, dataset='valid', title='checkpoint.pt'):
    #model.noise_std = args.noise_sigma
    model.eval()

    if dataset == 'train':
        sine_train_loader = train_dataset()
    else:
        sine_train_loader = valid_dataset()

    data = sine_train_loader[0]
    data = Variable(torch.Tensor(data))
    y_pred_clean = model(data)

    num_curves = len(data) if num_curves < 0 else num_curves

    plt.suptitle(title)

    for i in range(num_curves):
        plt.subplot(1, num_curves, i+1)

        cidx = i

        ref_y = sine_train_loader[0][cidx]
        ref_x = np.arange(len(ref_y))

        target_x = sine_train_loader[1][cidx]
        target_y = sine_train_loader[0][cidx][target_x]

        pred_x = y_pred_clean[cidx,-1].topk(1)[1].data.cpu().numpy()
        pred_y = sine_train_loader[0][cidx][pred_x]

        plt.plot(ref_x, ref_y)
        plt.plot(target_x, target_y, 'o')
        plt.plot(pred_x, pred_y, 'o')

        #plt.legend(['ref','target','pred'])

        print("Supervised")
        print(cidx, "Acc (sup):", (target_y == pred_y).mean())
        print(cidx, "MAE(pred,target)", abs(target_y - pred_y).mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    #parser.add_argument('--noise-sigma', type=float, default=0.3)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dataset', choices=['train','valid'], default='valid')
    parser.add_argument('--num-curves', type=int, default=-1)
    parser.add_argument('--checkpoint', default='model.pkl', type=str)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        model = torch.load(args.checkpoint)
        model.cuda()
    else:
        model = torch.load(args.checkpoint, map_location={'cuda:0':'cpu'})
        model.use_cuda = False

    plot(model,
         dataset=args.dataset,
         title=args.checkpoint,
         num_curves=args.num_curves)
    plt.show()
