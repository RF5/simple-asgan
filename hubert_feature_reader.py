# Copyright (c) Facebook, Inc. and its affiliates.
# Retrieved from https://github.com/pytorch/fairseq/blob/main/examples/textless_nlp/gslm/speech2unit/pretrained/hubert_feature_reader.py 
# Adapted by Matthew Baas

import torch
import fairseq
import soundfile as sf
import torch.nn.functional as F
from torch import Tensor

class HubertFeatureReader:
    """
    Wrapper class to run inference on HuBERT model.
    Helps extract features for a given audio file.
    """

    def __init__(self, checkpoint_path, layer: int, max_chunk=1600000, device='cuda', downsample_factor=320):
        """Instantiate a hubert feature reader from checkpoint file at `checkpoint_path`, setup to extract
        features from `layer` of the model. Optionally specify which `device` to perform inference on and the
        maximum chunk length to process audio on (chunk length is in samples).

        Note: all HuBERT fairseq models are trained on 16kHz audio, and this class assumes we are working with
        16kHz audio.
        """
        (
            model,
            _,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path]
        )
        self.model = model[0].eval().to(device)
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        self.device = device
        self.downsample_factor = downsample_factor

    def read_audio(self, path, ref_len=None):
        """ 
        Reads in audio from `path` using soundfile.
        Optionally ensure audio is of length `ref_len` samples. 
        """
        wav, sr = sf.read(path)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        assert sr == self.task.cfg.sample_rate, sr
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            print(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, file_path=None, ref_len=None, audio=None):
        """ 
        Converts audio in a given `file_path` to hubert features.
        Optionally directly provide `audio` as 1D float numpy array of correct sample rate.
        """
        if file_path is None: x = audio.float()
        elif audio is None: 
            x = self.read_audio(file_path, ref_len)
            x = torch.from_numpy(x).float().to(self.device)
        else: raise AssertionError("Either file_path or audio (not both) must be supplied.")
        
        with torch.no_grad():
            
            n_pad = self.downsample_factor - (x.shape[0] % self.downsample_factor)
            x = F.pad(x, (0, n_pad), value=0)
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)

    @torch.no_grad()
    def get_feats_batched(self, x: Tensor):
        """ Batched inference of embeddings. `x` is a waveform of shape (bs, T) """
        n_pad = self.downsample_factor - (x.shape[0] % self.downsample_factor)
        x = F.pad(x, (0, n_pad), value=0)

        if self.task.cfg.normalize: x = F.layer_norm(x, x.shape)

        feat = []
        for start in range(0, x.size(1), self.max_chunk):
            x_chunk = x[:, start: start + self.max_chunk]
            feat_chunk, _ = self.model.extract_features(
                source=x_chunk,
                padding_mask=None,
                mask=False,
                output_layer=self.layer,
            )
            feat.append(feat_chunk)
        return torch.cat(feat, 1)

