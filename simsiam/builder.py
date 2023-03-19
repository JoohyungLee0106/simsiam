# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from .resnet import resnet18, resnet34, resnet50, resnet101

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    FEAT = {'resnet18': {1: 64, 2: 64, 3: 128, 4: 256, 5: 512},
            'resnet34': {1: 64, 2: 64, 3: 128, 4: 256, 5: 512},
            'resnet50': {1: 64, 2: 256, 3: 512, 4: 1024, 5: 2048},
            'resnet101': {1: 64, 2: 256, 3: 512, 4: 1024, 5: 2048}}
    
    STRIDE = {'IMAGENET': {1: 4., 2: 4., 3: 8., 4: 16., 5: 32.},
              'CIFAR100': {1: 1., 2: 1., 3: 2., 4: 4., 5: 8.},
              'CIFAR10': {1: 1., 2: 1., 3: 2., 4: 4., 5: 8.}}
    
    def __init__(self, args, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = resnet50(dataset=args.dataset, layer_num=5, zero_init_residual=True, num_classes=1000 if args.dataset == 'IMAGENET' else 100, equiv_mode=args.equiv_mode)
        
        # build a 3-layer projector
        # prev_dim = self.encoder.fc.weight.shape[1]
        prev_dim = 2048
        self.projector_inv = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.projector_inv[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.projector_inv(torch.flatten(self.avgpool(self.encoder.forward_single(x1, layer_forward=5)), 1)) # NxC
        z2 = self.projector_inv(torch.flatten(self.avgpool(self.encoder.forward_single(x2, layer_forward=5)), 1)) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()
