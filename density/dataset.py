import logging
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from fastprogress.fastprogress import progress_bar
import random
import torchaudio
import math
from pathlib import Path

def _load(pth): 
    return (pth, torch.load(pth, map_location='cpu').half())

class LogMelSpectrogram(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.melspctrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=160,
            center=False,
            power=1.0,
            norm="slaney",
            onesided=True,
            n_mels=128,
            mel_scale="slaney",
        )

    def forward(self, wav):
        wav = F.pad(wav, ((1024 - 160) // 2, (1024 - 160) // 2), "reflect")
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        return logmel

class HubertFeatureDataset(Dataset):

    def __init__(self, filelist, zdim, mean=0, std=1, preload=False, data_type='hubert_L6') -> None:
        super().__init__()
        self.filelist = filelist
        self.data_type = data_type
        # self.z_dist = torch.distributions.Normal(mean, std, validate_args=True)
        self.z_dist = torch.distributions.MultivariateNormal(torch.zeros(zdim) + mean, 
                                            std*torch.eye(zdim), validate_args=True)
        self.preload = preload
        if data_type == 'melspec':
            self.mel_tfm = LogMelSpectrogram()
        if preload:
            print("Preloading dataset")
            from fastcore.parallel import parallel
            results = parallel(_load, filelist, n_workers=8, threadpool=True, progress=True)
            self.gt_path_dict = {}
            for pth, vec in results:
                self.gt_path_dict[pth] = vec

    def __getitem__(self, index) -> Tuple[Tensor]:
        gt_path = self.filelist[index]
        if self.preload: gt_feat = self.gt_path_dict[gt_path].float()
        else: 
            suf = Path(gt_path).suffix
            if suf in ['.wav', '.flac', '.mp3']:
                wav, sr_ = torchaudio.load(gt_path)
                if sr_ != 16000: raise AssertionError("Audio is not 16kHz!!!")
                if self.data_type == 'melspec':
                    mel = self.mel_tfm(wav) # (1, n_mels=128, N)
                    gt_feat = mel[0].T # (N, n_mels=c_dim)
                else: 
                    gt_feat = (0.99*wav/wav.max()).squeeze()
            elif suf in ['.pt', '.pth']:
                gt_feat = torch.load(gt_path, map_location='cpu') # (seq_len, c_dim)
            else: raise NotImplementedError()
        if self.data_type == 'wav':
            z = self.z_dist.sample((1,)).squeeze()
        else:
            z = self.z_dist.sample((gt_feat.shape[0],)).squeeze()
        return gt_feat, z

    def __len__(self): return len(self.filelist)

    def prep_fad_metrics(self, hubert=None, hifigan=None):
        print("Computing FAD metrics")
        if self.preload == False:
            if self.data_type == 'melspec':
                print("FAD metric from mel-spectrograms. Will take a while to compute on CPU...")
                items = [self.__getitem__(i)[0] for i in progress_bar(range(len(self)))]
                ml = max([t.shape[0] for t in items])
                for i in progress_bar(range(len(self))):
                    it = items[i]
                    n_pad = max(0, ml - it.shape[0])
                    it = F.pad(it, (0, 0, n_pad, 0), value=-11.4)
                    items[i] = it
                data = torch.stack(items, dim=0).float()
                # convert each melspec --> audio --> hubert embedding
                n_chunk = math.ceil(data.shape[0] / 32)
                chunks = data.chunk(n_chunk, dim=0)
                feats = []
                with torch.no_grad():
                    for c in progress_bar(chunks):
                        wavr, sr = hifigan.generate(c.permute(0, 2, 1)) # needs (bs, n_mels, N)
                        # wavr is of shape (bs, 1, T)
                        feat = hubert.get_feats_batched(wavr.squeeze(1))# (bs, T, c_dim)
                        feats.append(torch.flatten(feat, start_dim=0, end_dim=1))
                del data
                del items
                data = torch.cat(feats, dim=0)
            elif self.data_type == 'wav':
                print("FAD metric from wav. Will take a while to compute on CPU...")
                items = [self.__getitem__(i)[0] for i in progress_bar(range(len(self)))]
                ml = max([t.shape[0] for t in items])
                for i in progress_bar(range(len(self))):
                    it = items[i]
                    n_pad = max(0, ml - it.shape[0])
                    if n_pad > 0:
                        it = F.pad(it, (0, n_pad), value=0)
                    items[i] = it
                data = torch.stack(items, dim=0).float()
                # convert each audio --> hubert embedding
                n_chunk = math.ceil(data.shape[0] / 32)
                chunks = data.chunk(n_chunk, dim=0)
                feats = []
                with torch.no_grad():
                    for wavr in progress_bar(chunks):
                        feat = hubert.get_feats_batched(wavr)# (bs, T, c_dim)
                        feats.append(torch.flatten(feat, start_dim=0, end_dim=1))
                del data
                del items
                data = torch.cat(feats, dim=0)
            else:
                data = torch.cat([self.__getitem__(i)[0] for i in progress_bar(range(len(self)))], dim=0).float()
        else:
            data = torch.cat(list(self.gt_path_dict.values()), dim=0).float()
        self.mean = data.mean(dim=0)
        self.cov = data.T.cov()
        del data


def rv_collate(xs):
    z = torch.cat([x[1] for x in xs], dim=0)
    gt_feats = torch.cat([x[0] for x in xs], dim=0)
    return gt_feats, z

class RP_Collate():

    def __init__(self, c_dim, seq_len, data_type='hubert_L6', pad_type='tile') -> None:
        super().__init__()
        self.data_type = data_type
        try:
            if self.data_type == 'hubert_L6':
                self.sil_vec = torch.load('density/runs/hubert_sil_vec.pt')
            elif self.data_type == 'melspec':
                self.sil_vec = torch.ones((c_dim,), dtype=torch.float)*(-11.4)
            elif self.data_type == 'wav':
                self.sil_vec = torch.tensor(0)
            else: raise NotImplementedError()
        except Exception:
            logging.warning("[DATASET] Failed to fetch silence vector. Using zero silence vector.")
            self.sil_vec = None
        self.c_dim = c_dim
        self.seq_len = seq_len
        self.pad_type = pad_type
    
    def __call__(self, xs):
        # def rp_collate(xs, seq_len):
        seq_len = self.seq_len
        feat_dim = xs[0][0].shape[-1]
        if self.sil_vec is None:
            gt_feats = torch.zeros(len(xs), seq_len, feat_dim, dtype=torch.float32)
        else:
            if self.data_type == 'wav': gt_feats = self.sil_vec[None, None].repeat(len(xs), seq_len).to(torch.float32)
            else: gt_feats = self.sil_vec[None, None].repeat(len(xs), seq_len, 1).to(torch.float32)
        next_feats = []
        lengths = []
        for i in range(len(xs)):
            l = xs[i][0].shape[0]
            if l <= seq_len: 
                lengths.append(l)
                gt_feats[i, :l] = xs[i][0][:l]
                if self.pad_type == 'tile':
                    if seq_len - l < l: # ensure that we can tile once and once only
                        gt_feats[i, l:] = xs[i][0][:seq_len - l]

            else:
                lengths.append(seq_len)
                start_ind = random.randint(0, l - seq_len - 1)
                gt_feats[i] = xs[i][0][start_ind:start_ind+seq_len]
        
        lengths = torch.tensor(lengths).to(torch.long)
        # Remove next feats, we are not training a LM here.
        if self.data_type == 'wav':
            z = torch.stack([x[1] for x in xs], dim=0)
        else:
            z = torch.stack([x[1][0] for x in xs], dim=0)
        return gt_feats, lengths, None, z


