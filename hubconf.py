dependencies = ['torch', 'torchaudio', 'numpy', 'omegaconf']

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlparse 

from omegaconf import OmegaConf
from density.models import RP_W


class Conv2SC09Wrapper(nn.Module):

    def __init__(self, g: nn.Module, hifigan: nn.Module, cfg, seq_len=16000) -> None:
        """ Inference wrapper for generator `g` and `hifigan` vocoder for the 
        conv2 series of models which predict a sequence of hubert vectors, which is then 
        vocoded with a hifigan trained to convert hubert vectors to waveforms.
        """
        super().__init__()
        self.g = g
        self.z_dim = self.g.z_dim
        self.w_dim = self.g.w_dim
        self.hifigan = hifigan
        self.cfg = cfg
        self.seq_len = seq_len

    def unconditional_generate(self, N: int):
        """ Generate `N` audio samples, returning a tensor of shape (N, 16000) """
        z = torch.randn(N, self.g.z_dim).to(self.g.preconv.bias.device)
            
        c = self.g(z, self.cfg.rp_w_cfg.seq_len)
        audio = self.hifigan(c).squeeze(1)

        n_pad = self.seq_len - audio.shape[-1]
        if n_pad > 0:
            audio = F.pad(audio, (n_pad//2, n_pad - n_pad//2), value=0)
        return audio

    def generate_from_latent(self, z: Tensor) -> Tensor:
        """ Generate waveforms (N, 16000) from latent standard normal `z` (N, z_dim) """
        z = z.to(self.g.preconv.bias.device)
                    
        c = self.g(z, self.cfg.rp_w_cfg.seq_len)
        audio = self.hifigan(c).squeeze(1)

        n_pad = self.seq_len - audio.shape[-1]
        if n_pad > 0:
            audio = F.pad(audio, (n_pad//2, n_pad - n_pad//2), value=0)

        return audio

    def z2w(self, z: Tensor) -> Tensor:
        """ Generate latent W vectors (N, w_dim) from latent standard normal `z` (N, z_dim) """
        z = z.to(self.g.preconv.bias.device)
        w = self.g.W(z) 
        return w

    def generate_from_w(self, w: Tensor) -> Tensor:
        """ Generate waveforms (N, 16000) from W latent space `w` (N, w_dim) """
        w = w.to(self.g.preconv.bias.device)

        c = self.g.forward_w(w, self.cfg.rp_w_cfg.seq_len)
        audio = self.hifigan(c).squeeze(1)

        n_pad = self.seq_len - audio.shape[-1]
        if n_pad > 0:
            audio = F.pad(audio, (n_pad//2, n_pad - n_pad//2), value=0)

        return audio


def asgan_hubert_sc09_6(pretrianed=True, progress=True, device='cuda'):
    """ Density GAN which generates mel-spectrograms from standard normal vectors and then 
    uses hifigan to vocode them back to the time domain.
    """
    if torch.cuda.is_available() == False:
        if str(device) != 'cpu':
            logging.warning(f"Overriding device {device} to cpu since no GPU is available.")
            device = 'cpu'

    ckpt = torch.hub.load_state_dict_from_url(
        "https://github.com/RF5/simple-asgan/releases/download/v0/asgan_hubert_ckpt_00520000_slim.pt",
        map_location=device,
        progress=progress
    )
    cfg = OmegaConf.create(ckpt['cfg_yaml'])
    cfg.device = 'cpu'

    if cfg.data_type == 'melspec':
        hifigan = torch.hub.load("bshall/hifigan:main", "hifigan", map_location='cpu').cpu().eval()
    else:
        hifigan = hubert_hifigan(progress=progress, pretrained=pretrianed, device='cpu')
    print(f"HiFiGAN loaded with {sum([p.numel() for p in hifigan.parameters()]):,d} parameters.")

    g = RP_W(cfg.rp_w_cfg).cpu().eval()
    if pretrianed:
        g.load_state_dict(ckpt['g_ema_state_dict'])
    
    model = Conv2SC09Wrapper(g, hifigan, cfg)
    model = model.to(device).eval()
    return model

def hubert_hifigan(pretrained=True, progress=True, device='cuda', model_dir=None):
    """ HiFiGAN which works on HuBERT embeddings. Optionally specify `model_dir` as location of hubert checkpoint. """
    if not pretrained: raise NotImplementedError("Only pretrained model supported.")

    if model_dir is None:
        hub_dir = torch.hub.get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')
    url = "https://github.com/RF5/simple-asgan/releases/download/v0/g_02365000_package.pth"
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        torch.hub.download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    pi = torch.package.package_importer.PackageImporter(cached_file)
    hifigan = pi.load_pickle('.', 'hifigan_pkl.pt', map_location=device)
    hifigan.eval()
    logging.info(f"[MODEL] HubertHiFiGAN loaded with {sum([p.numel() for p in hifigan.parameters()]):,d} parameters")
    del pi
    return hifigan

def hubert_base(pretrained=True, progress=True, device='cuda'):
    """ Facebook HuBERT BASE model"""
    if not pretrained: raise NotImplementedError()
    _ = torch.hub.load_state_dict_from_url(
        "https://github.com/RF5/simple-asgan/releases/download/v0/hubert_base_ls960_nofinetune.pt",
        map_location='cpu', progress=progress
    )
    base_path = str(Path(torch.hub.get_dir())/'checkpoints'/'hubert_base_ls960_nofinetune.pt')

    from hubert_feature_reader import HubertFeatureReader
    hubert = HubertFeatureReader(base_path, 6, device=device)
    logging.info(f"[MODEL] HuBERT loaded with {sum([p.numel() for p in hubert.model.parameters()]):,d} parameters")
    return hubert
    