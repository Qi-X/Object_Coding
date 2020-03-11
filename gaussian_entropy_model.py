# -*- coding: UTF-8 -*-
import torch
import numpy as np
import torch.nn as nn


## make probability positive, if a < 10e-6, make gradient inverse
class Low_bound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):

        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-6)
        return x
    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        grad1[x<1e-6] = 0
        pass_through_if = np.logical_or(x.cpu() >= 1e-6,g.cpu()<0.0)
        t = pass_through_if.float().cuda()


        return grad1 * t


class Distribution_for_entropy(nn.Module):
    def __init__(self):
        super(Distribution_for_entropy,self).__init__()


    def forward(self, x, p_dec):
        _,C,_,_ = x.size()
        #print (x.size,p_dec.size())
        mean = p_dec[:,:C, :, :]
        scale= p_dec[:,C:, :, :]

    ## to make the scale always positive
        scale[scale == 0] = 1e-9

        m1 = torch.distributions.normal.Normal(mean,scale)
        #m1 = torch.distributions.laplace.Laplace(mean, scale)
        lower = m1.cdf(x - 0.5)
        upper = m1.cdf(x + 0.5)


        likelihood = torch.abs(upper - lower)

        likelihood = Low_bound.apply(likelihood)
        return likelihood

class Gaussian_entropy_model(nn.Module):
    def __init__(self):
        super(Gaussian_entropy_model,self).__init__()

    def forward(self, x, p_dec):
        mean = p_dec[:, 0,:, :, :]
        scale= p_dec[:, 1,:, :, :]
        scale[scale == 0] = 1e-6
        m1 = torch.distributions.normal.Normal(mean,scale)
        lower = m1.cdf(x - 0.5)
        upper = m1.cdf(x + 0.5)
        likelihood = torch.abs(upper - lower)
        likelihood = Low_bound.apply(likelihood)

        return likelihood

