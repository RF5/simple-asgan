from typing import Tuple
import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
import matplotlib.pyplot as plt
import logging
import torch.nn as nn
import librosa.display

class ApproxKL():

    def __init__(self, bins=200, cmin=-1.2, cmax=1.2) -> None:
        """ 
        Approximates a KL divergence metric between 1D distributions of values from 
        real and fake c embeddings. 
        """
        self.bins = torch.linspace(cmin, cmax, bins)
        self.clear()

    def accumulate(self, c_real: Tensor, c_fake: Tensor) -> None:
        """ `c_real` and `c_fake` of shape (bs, dim) """
        bs, dim = c_real.shape
        real_cnts, _ = torch.histogram(c_real.cpu().clamp(self.bins.min(), self.bins.max()), F.pad(self.bins, (0, 1), value=self.bins[-1]+1))
        fake_cnts, _ = torch.histogram(c_fake.cpu().clamp(self.bins.min(), self.bins.max()), F.pad(self.bins, (0, 1), value=self.bins[-1]+1))
        self.real_cnts += real_cnts
        self.fake_cnts += fake_cnts


    def metric(self):
        real_dist = (self.real_cnts / self.real_cnts.sum()).clamp_(1e-7)
        fake_dist = (self.fake_cnts / self.fake_cnts.sum()).clamp_(1e-7)
        return F.kl_div(real_dist.log(), fake_dist.log(), log_target=True, reduction='batchmean')

    def clear(self):
        self.real_cnts = torch.zeros_like(self.bins)
        self.fake_cnts = torch.zeros_like(self.bins)

    def make_plot(self):
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
        axs[0].set_title("Histogram of real counts")
        axs[0].bar(self.bins.numpy(), (self.real_cnts / self.real_cnts.sum()).numpy())
        axs[1].set_title("Histogram of fake counts")
        axs[1].bar(self.bins.numpy(), (self.fake_cnts / self.fake_cnts.sum()).numpy())
        fig.tight_layout()
        return fig

def closest_dist(c_real, c_fake, reduce='mean', n1=50, n2=2500) -> Tensor:
    # (bs, dim)
    inds = torch.randperm(c_real.shape[0])[:n2]
    # dists = ((c_real[None, inds] - c_fake[:n1, None])**2).sum(dim=-1) # (n1, n2)
    dists = 1 - F.cosine_similarity(c_real[None, inds], c_fake[:n1, None], dim=-1) # (n1, n2)
    min_dists = dists.min(dim=1).values # (n1,)
    return min_dists.mean() if reduce=='mean' else min_dists

def closest_dist_pair(c_real, c_fake, n1=1000, n2=1000) -> Tuple[Tensor, Tensor]:
    inds = torch.randperm(c_real.shape[0])[:n2]
    inds2 = torch.randperm(c_fake.shape[0])[:n1]
    # dists = ((c_real[None, inds] - c_fake[:n1, None])**2).sum(dim=-1) # (bs, bs)
    dists = 1 - F.cosine_similarity(c_real[None, inds].cpu(), c_fake[inds2, None].cpu(), dim=-1) # (bs, bs)
    min_dists = dists.min(dim=1).values # (n1)
    min_dists2 = dists.min(dim=0).values # (n2)
    return min_dists.float(), min_dists2.float()

@torch.no_grad()
def perc_closest(real: Tensor, fake: Tensor, closest_n: int = 100, n=1000) -> Tuple[Tensor]:
    """ Find % of `closest_n` that belong to `real` or `fake` (bs, c_dim) cs"""
    inds = torch.randperm(real.shape[0])[:n]
    inds2 = torch.randperm(fake.shape[0])[:n]
    inner_real_dists = 1 - F.cosine_similarity(real[None, inds], real[inds, None], dim=-1) # (n, n)
    inner_real_dists = inner_real_dists.topk(closest_n, dim=-1, largest=False).values # (n, 100)

    cross_dists = 1 - F.cosine_similarity(real[None, inds], fake[inds2, None], dim=-1) # (n, n)
    cross_dists1 = cross_dists.topk(closest_n, dim=-1, largest=False).values # (n, 100)
    cross_dists2 = cross_dists.topk(closest_n, dim=0, largest=False).values # (n, 100)
    del cross_dists

    inner_fake_dists = 1 - F.cosine_similarity(fake[None, inds2], fake[inds2, None], dim=-1)
    inner_fake_dists = inner_fake_dists.topk(closest_n, dim=-1, largest=False).values # (n, 100)
    # Find perc for % fake closest to real
    real_dists = torch.cat((inner_real_dists, cross_dists2.permute(1, 0)), dim=-1)
    ii = real_dists.argsort(dim=-1)[:, :closest_n]
    fake_closest_to_real_perc = (ii > closest_n).sum(dim=-1)/(closest_n)
    # Find perc of % real closest to fake
    fake_dists = torch.cat((inner_fake_dists, cross_dists1), dim=-1)
    ii2 = fake_dists.argsort(dim=-1)[:, :closest_n]
    real_closest_to_fake_perc = (ii2 > closest_n).sum(dim=-1)/(closest_n)

    return fake_closest_to_real_perc.mean(), real_closest_to_fake_perc.mean()


def fad(c_f, mu_real, cov_real):
    mu_fake = c_f.mean(dim=0)
    cov_fake = c_f.T.double().cov().float()
    fad = calculate_frechet_distance(mu_fake.numpy(), cov_fake.numpy(), mu_real.numpy(), cov_real.numpy())
    return fad

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.

    Retrieved and adapted from 
    https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    """
    from scipy.linalg import sqrtm
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.astype(np.float64).dot(sigma2.astype(np.float64)), disp=False)
    # covmean = covmean.astype(np.float32)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            logging.warning(f"Imaginary component {m} in FAD calculation. Returning NaN.")
            return float('nan')
            #raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def make_grad_norm_fig(g: nn.Module, d: nn.Module):
    # Make generator fig:
    plot_data = []
    for nm, p in g.named_parameters():
        gnorm = float(torch.linalg.norm(p.grad)) if p.grad is not None else float('nan')
        plot_data.append((nm, gnorm))
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 16))
    # ax.grid(True)
    axs[0].barh([p[0] for p in plot_data], [p[1] for p in plot_data])
    for i, tick in enumerate(axs[0].yaxis.get_ticklabels()):
        if np.isnan(plot_data[i][1]):
            tick.set_color('red')
    axs[0].set_xlabel("Flattened gradient L2 norm")
    axs[0].set_title("L2 norm of Generator param gradients")

    plot_data2 = []
    for nm, p in d.named_parameters():
        plot_data2.append((nm, float(torch.linalg.norm(p.grad)) if p.grad is not None else float('nan')))
    axs[1].barh([p[0] for p in plot_data2], [p[1] for p in plot_data2])
    for i, tick in enumerate(axs[1].yaxis.get_ticklabels()):
        if np.isnan(plot_data2[i][1]):
            tick.set_color('red')
    axs[1].set_xlabel("Flattened gradient L2 norm (red=nan)")
    axs[1].set_title("L2 Norm of Discriminator param gradients")
    fig.tight_layout()
    return fig

def plot_melspecs(c_fake: Tensor, n=8, is_audio=False):
    """ `c_fake` (bs, seq_len, c_dim) """
    if is_audio:
        from density.dataset import LogMelSpectrogram
        c_fake = LogMelSpectrogram().forward(c_fake) # (1, n_mels=128, N)
        c_fake = c_fake.permute(0, 2, 1)

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(15, 7))
    axs = axs.flatten()
    for i in range(min(n, c_fake.shape[0])):
        librosa.display.specshow(c_fake[i].T.cpu().numpy(), cmap='viridis', ax=axs[i], vmin=-11.4, vmax=2.5)
    fig.tight_layout()
    return fig
