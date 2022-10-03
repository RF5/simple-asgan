from typing import Tuple
import torch.nn as nn
import torch 
from torch import Tensor
import math
import numpy as np
import random
from fastprogress import progress_bar
import logging
import torch.nn.functional as F


class Augmenter():
    def __init__(self, noise_std=0.05, n=2, p=0.05, ADA_N=6, ADA_adjust=0.005) -> None:
        self.noise_std = noise_std
        self.n = n 
        self.n_dist = torch.distributions.Categorical(torch.as_tensor([1/(2*n+1)]*(2*n+1)))
        self.p = p
        self.running_d = 0.0
        self.batch_cnt = 0
        self.expecation_count = 0
        self.ADA_N = ADA_N
        self.ADA_adjust = ADA_adjust
        self.rt = 0.6
        logging.info(f"[AUGMENT] Noise std set at {noise_std}")

    def __call__(self, c_real: Tensor, c_fake: Tensor) -> Tuple[Tensor, Tensor]:
        """ `c` is of shape (bs, seq_len, c_dim) """
        ps = torch.bernoulli(torch.empty(5,).fill_(self.p)).bool()
        bs, seq_len, _ = c_real.shape
        # Add noise
        if ps[0]:
            c_real = c_real.add(torch.randn_like(c_real)*self.noise_std)
            c_fake = c_fake.add(torch.randn_like(c_fake)*self.noise_std)

        # Random scale
        if ps[2]:
            scale = 1 + 0.05*torch.randn((bs, 2), device=c_real.device)
            c_real = c_real*(scale[:, 0, None, None])
            c_fake = c_fake*(scale[:, 1, None, None])
        # Swap a frame of real for fake
        if ps[3]:
            swap_inds = torch.randint(0, seq_len, (bs,))
            a = torch.arange(bs)
            c_real[a, swap_inds].copy_(c_fake[a, swap_inds].detach())
        # Real-as-fake v2
        if ps[4]:
            lens = torch.randint(0, 1 + seq_len//2, (bs, ))
            
            start_inds1 = torch.tensor([random.randint(0, seq_len-l-1) for l in lens]).long()
            start_inds2 = torch.tensor([random.randint(0, seq_len-l-1) for l in lens]).long()
            c_fake2 = c_fake.detach()
            r = torch.rand((bs, 2))
            r1 = r[:, 0]
            r2 = r[:, 1]
            
            lens2 = torch.randint(0, 1 + seq_len//2, (bs, ))
            start_inds1v2 = torch.tensor([random.randint(0, seq_len-l-1) for l in lens2]).long()
            start_inds2v2 = torch.tensor([random.randint(0, seq_len-l-1) for l in lens2]).long()
            for i in range(bs):
                c_real[i, start_inds1[i]:start_inds1[i]+lens[i]] = (1-r1[i])*c_real[i, start_inds1[i]:start_inds1[i]+lens[i]] \
                                    + r1[i]*c_fake2[i, start_inds2[i]:start_inds2[i]+lens[i]]

                c_fake[i, start_inds1v2[i]:start_inds1v2[i]+lens2[i]] = (1-r2[i])*c_real[i, start_inds1v2[i]:start_inds1v2[i]+lens2[i]] \
                                    + r2[i]*c_fake2[i, start_inds2v2[i]:start_inds2v2[i]+lens2[i]]

        return c_real, c_fake

    def skip_d_update(self) -> bool:
        return random.random() < self.p

    def adjust_d_lr(self, old_lr: float, clamp_min=3e-8, clamp_max=5e-3) -> float:
        if self.rt < 0.6:
            # adjust lr up (D gets stronger)
            new_lr = min(old_lr*1.01, clamp_max)
        else:
            # adjust lr down (D gets weaker)
            new_lr = max(old_lr*0.99, clamp_min)
        return new_lr

    @torch.no_grad()
    def accumulate(self, d_real_outs: Tensor) -> None:
        """ Homebrew ADA accumulation """
        self.batch_cnt += 1
        self.expecation_count += d_real_outs.numel()
        self.running_d += torch.sign(d_real_outs).sum().cpu()
        if self.batch_cnt == self.ADA_N:
            self.rt = self.running_d/self.expecation_count
            if self.rt > 0.98:
                self.p += 10*self.ADA_adjust
            elif self.rt > 0.6: # as per ADA paper
                # increase p 
                self.p += self.ADA_adjust
            elif self.rt < 0.6:
                self.p -= self.ADA_adjust
            self.p = np.clip(self.p, 1e-8, 1-1e-8)

            self.batch_cnt = 0
            self.expecation_count = 0
            self.running_d = 0.0


def melspec2hubert(cs: Tensor, hubert, hifigan, bs=32, mb=None) -> Tensor:
    cs = cs.cpu()
    n_chunk = math.ceil(cs.shape[0] / bs)
    chunks = cs.chunk(n_chunk, dim=0)
    feats = []
    with torch.no_grad():
        pb = progress_bar(chunks, total=len(chunks), parent=mb)
        for c in pb:
            wavr, sr = hifigan.generate(c.permute(0, 2, 1)) # needs (bs, n_mels, N)
            # wavr is of shape (bs, 1, T)
            feat = hubert.get_feats_batched(wavr.squeeze(1))# (bs, T, c_dim)
            feats.append(feat)
    feats = torch.cat(feats, dim=0) # (c.shape[0], seq_len, c_dim)
    return feats

def wav2hubert(cs: Tensor, hubert, bs=32, mb=None) -> Tensor:
    cs = cs.cpu()
    n_chunk = math.ceil(cs.shape[0] / bs)
    chunks = cs.chunk(n_chunk, dim=0)
    feats = []
    with torch.no_grad():
        pb = progress_bar(chunks, total=len(chunks), parent=mb)
        for c in pb:
            feat = hubert.get_feats_batched(c)# (bs, T, c_dim)
            feats.append(feat)
    feats = torch.cat(feats, dim=0) # (c.shape[0], seq_len, c_dim)
    return feats