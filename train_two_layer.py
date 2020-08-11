#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import datetime
import pathlib
import random
import json
import numpy as np

import torch

import sys
sys.path.append('./code/')
from linear_utils import linear_model

"""
A fully-connected ReLU network with one hidden layer, trained to predict y from x
by minimizing squared Euclidean distance.

This implementation uses the nn package from PyTorch to build the network.
PyTorch autograd makes it easy to define computational graphs and take gradients,
but raw autograd can be a bit too low-level for defining complex neural networks;
this is where the nn package can help. The nn package defines a set of Modules,
which you can think of as a neural network layer that has produces output from
input and may have some trainable weights or other state.
"""

# get CLI parameters
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', type=str, default='', metavar='FILE',
                    help='JSON file containing the configuration dictionary')

parser = argparse.ArgumentParser(description='CLI parameters for training')
parser.add_argument('--root', type=str, default='', metavar='DIR',
                    help='Root directory')
parser.add_argument('-t', '--iterations', type=int, default=1e4, metavar='ITERATIONS'
                    help='Iterations (default: 1e4)')
parser.add_argument('-n', '--samples', type=int, default=100, metavar='N'
                    help='Number of samples (default: 100)')
parser.add_argument('--print-freq', type=int, default=1000,
                    help='CLI output printing frequency (default: 1000)')
parser.add_argument('--gpu', type=int, default=None,
                    help='Number of GPUS to use')
parser.add_argument('--seed', type=int, default=None,
                    help='Random seed')                        
parser.add_argument('-d', , '--dim', type=int, default=50, metavar='DIMENSION'
                    help='Feature dimension (default: 50)')
parser.add_argument('--hidden', type=int, default=200, metavar='DIMENSION'
                    help='Hidden layer dimension (default: 200)')
# parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
#                     help='Optimizer (default: "sgd")')
# parser.add_argument('--loss', type=str, default='cross_entropy', metavar='LOSS',
#                     help='loss function (default: cross entropy')
# parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
#                     help='SGD momentum (default: 0.9)')                        
# parser.add_argument('--weight-decay', type=float, default=0.0001,
#                     help='weight decay (default: 0.0001)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')                 
parser.add_argument('--use-inverse-sqrt-lr', action='store_true', default=False,
                    help='Use inverse square-root learning rate decay')
parser.add_argument('--details', type=str, metavar='N',
                    default='no_detail_given',
                    help='details about the experimental setup')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config) as f:
            cfg = json.load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    return args

# set parameters
args = _parse_args()

# directories
root = pathlib.Path(args.root) if args.root else pathlib.Path.cwd()

current_date = str(datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
args.outpath = (pathlib.Path.cwd() / 'results' / 'two_layer_nn' /  current_date)

args.outpath.mkdir(exist_ok=True, parents=True)

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    warnings.warn('You have chosen to seed training. '
                'This will turn on the CUDNN deterministic setting, '
                'which can slow down your training considerably! '
                'You may see unexpected behavior when restarting '
                'from checkpoints.')

if args.gpu is not None:
    warnings.warn('You have chosen a specific GPU. This will completely '
                'disable data parallelism.')


device = torch.device('cpu')
# device = torch.device('cuda') # Uncomment this to run on GPU

d_out = 1
# Create random Tensors to hold inputs and outputs
lin_model = linear_model(args.dim, sigma_noise=0.0, normalized=False)
Xs, ys = lin_model.sample(args.samples)
Xs = torch.Tensor(Xs).to(device)
ys = torch.Tensor(ys).to(device)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
# After constructing the model we use the .to() method to move it to the
# desired device.
model = torch.nn.Sequential(
          torch.nn.Linear(args.dim, args.hidden),
          torch.nn.ReLU(),
          torch.nn.Linear(args.hidden, d_out),
        ).to(device)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function. Setting
# reduction='sum' means that we are computing the *sum* of squared errors rather
# than the mean; this is for consistency with the examples above where we
# manually compute the loss, but in practice it is more common to use mean
# squared error as a loss by setting reduction='elementwise_mean'.
loss_fn = torch.nn.MSELoss(reduction='sum')

losses = []
for t in range(args.iterations):
  # Forward pass: compute predicted y by passing x to the model. Module objects
  # override the __call__ operator so you can call them like functions. When
  # doing so you pass a Tensor of input data to the Module and it produces
  # a Tensor of output data.
  y_pred = model(Xs)

  # Compute and print loss. We pass Tensors containing the predicted and true
  # values of y, and the loss function returns a Tensor containing the loss.
  loss = loss_fn(y_pred, ys)
  losses.append(loss.item())
  print(t, loss.item())
  
  # Zero the gradients before running the backward pass.
  model.zero_grad()

  # Backward pass: compute gradient of the loss with respect to all the learnable
  # parameters of the model. Internally, the parameters of each Module are stored
  # in Tensors with requires_grad=True, so this call will compute gradients for
  # all learnable parameters in the model.
  loss.backward()

  # Update the weights using gradient descent. Each parameter is a Tensor, so
  # we can access its data and gradients like we did before.
  with torch.no_grad():
    for param in model.parameters():
      param.data -= learning_rate * param.grad

np.save(args.outpath / 'loss.npy', np.array(losses))