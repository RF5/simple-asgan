import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import math

def logistic_g_loss(d_generated_outputs: Tensor) -> Tensor:
    """
    Logistic generator loss.
    Assumes input is D(G(x)), or in our case, D(W(z)).
    `disc_outputs` of shape (bs,)
    """
    # d_generated_outputs = torch.sigmoid(d_generated_outputs)
    # loss = torch.log(1 - d_generated_outputs).mean()
    loss = F.softplus(-d_generated_outputs).mean()
    return loss

def logistic_d_loss(d_real_outputs, d_generated_outputs):
    """
    Logistic discriminator loss.
    `d_real_outputs` (bs,): D(x), or in our case D(c)
    `d_generated_outputs` (bs,): D(G(x)), or in our case D(W(z))
    D attempts to push real samples as big as possible (as close to 1.0 as possible), 
    and push fake ones to 0.0
    """
    # d_real_outputs = torch.sigmoid(d_real_outputs)
    # d_generated_outputs = torch.sigmoid(d_generated_outputs)
    # loss = -( torch.log(d_real_outputs) + torch.log(1-d_generated_outputs) )
    # loss = loss.mean()
    term1 = F.softplus(d_generated_outputs) 
    term2 = F.softplus(-d_real_outputs)
    return (term1 + term2).mean()


def dist_heuristic_loss(c_real:  Tensor, c_fake: Tensor, n) -> Tensor:
    inds = torch.randperm(c_fake.shape[0])[:n]

    inner_real_dists = 1 - F.cosine_similarity(c_real[None, inds], c_real[inds, None], dim=-1) # (n, n)
    # inner_real_dists = inner_real_dists.topk(closest_n, dim=-1, largest=False).values # (n, 100)
    mean_real_dist = inner_real_dists.mean()

    cross_dists = 1 - F.cosine_similarity(c_real[None, inds], c_fake[inds, None], dim=-1) # (n, n)
    # cross_dists1 = cross_dists.topk(closest_n, dim=-1, largest=False).values # (n, 100)
    # cross_dists2 = cross_dists.topk(closest_n, dim=0, largest=False).values # (n, 100)
    mean_cross_dist = cross_dists.mean()
    
    inner_fake_dists = 1 - F.cosine_similarity(c_fake[None, inds], c_fake[inds, None], dim=-1)
    # inner_fake_dists = inner_fake_dists.topk(closest_n, dim=-1, largest=False).values # (n, 100)
    mean_fake_dist = inner_fake_dists.mean()

    # we want to make the mean real distances as close as possible to the mean fake and mean cross dists
    loss = (mean_fake_dist - mean_real_dist)**2 + (mean_cross_dist - mean_real_dist)**2
    return loss

def r1_reg(d_real_outputs: Tensor, c_real: Tensor, r1_gamma) -> Tensor:
    # r1_grads is of same shape as c_real, i.e. (bs, seq_len, c_dim)
    r1_grads = torch.autograd.grad(outputs=[d_real_outputs.sum()], inputs=[c_real], create_graph=True)[0]
    # print(r1_grads.shape, r1_grads) 
    r1_loss = r1_grads.norm(p=2, dim=-1).square() # || \Delta D(c_real) ||^2
    r1_loss = r1_loss.sum() # E[ || \Delta D(c_real) ||^2 ]
    r1_loss = (r1_gamma/2)*r1_loss

    return r1_loss
