import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import logging

FIX_DISCRIMINATOR_FIRST_BLOCK = True


class RV_W(nn.Module):
    """ Maps z to c as a random variable. i.e. this models the function W(z) = c"""

    def __init__(self, z_dim, c_dim, n_layers=3, leaky_alpha=0.1, equalized_lr=False, equalized_lr_mult=0.01) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.n_layers = n_layers
        layers = [nn.Linear(self.z_dim, self.c_dim), nn.LeakyReLU(leaky_alpha)]
        for n in range(n_layers):
            layers.extend([nn.Linear(self.c_dim, self.c_dim), nn.LeakyReLU(leaky_alpha) if n != n_layers - 1 else nn.Identity()])
        
        self.equalized_lr = equalized_lr
        offset = 1 if z_dim != c_dim else 0
        if equalized_lr:
            self.fcs = nn.ModuleList(layers)
            offset = 0
        else:
            self.fcs = nn.Sequential(*layers)
        for i, l in enumerate(layers[offset:]):
            if hasattr(l, 'bias'): 
                nn.init.zeros_(l.bias)
                if equalized_lr:
                    nn.init.normal_(l.weight, std=1/equalized_lr_mult)
                    weight_gain = equalized_lr_mult / math.sqrt(l.in_features)
                    bias_gain = equalized_lr_mult
                    setattr(self, f"layer_{i}_scale", weight_gain)
                    setattr(self, f"layer_{i}_bias_scale", bias_gain)
                else:
                    l.weight.data += torch.eye(l.weight.data.shape[0], dtype=l.weight.dtype)

    def forward(self, z: Tensor) -> Tensor:
        """ Assumes `z` of shape (bs, z_dim) """
        if self.equalized_lr:
            c = z
            for i, l in enumerate(self.fcs):
                if type(l) == nn.Linear:
                    ws = getattr(self, f'layer_{i}_scale')
                    bs = getattr(self, f'layer_{i}_bias_scale')
                    c = F.linear(c, ws*l.weight, bs*l.bias)
                else:
                    c = l(c)
        else:
            c = self.fcs(z)
        return c

    def extra_repr(self):
        return (f"layers={self.n_layers:d}, equalized_lr={self.equalized_lr}")


class RP_W(nn.Module):
    """ Maps z to c as a random process. i.e. this models the function W(z) = {c_1, c_2, c_3...} """

    def __init__(self, cfg) -> None:
        from .config import RP_W_Config
        super().__init__()
        self.cfg: RP_W_Config = cfg
        
        self.z_dim = self.cfg.z_dim
        self.c_dim = self.cfg.c_dim
        self.w_dim = self.cfg.w_dim
        self.w_layers = self.cfg.w_layers
        self.W = RV_W(self.z_dim, self.w_dim, self.w_layers, cfg.lrelu_coeff, cfg.equalized_lr, cfg.w_lr_mult)

        self.first_seq_len = int(self.cfg.layer_specs[0][2])
        if cfg.use_sg3_ff:
            self.ff = SynthesisInput(self.w_dim, self.cfg.ff_channels, self.first_seq_len,
                                     cfg.equalized_lr)
        else:
            self.ff = FourierFeature(self.w_dim, self.cfg.ff_channels, 
                                    self.cfg.init_B_std, cfg.equalized_lr)
        # Funny story: stylegan 3 claims to use this 1x1 conv, but it never fucking uses it in their 
        # code implementation. 
        self.preconv = nn.Conv1d(self.cfg.ff_channels, self.cfg.ff_channels, 1, 1)
        nn.init.zeros_(self.preconv.bias)
        if cfg.equalized_lr:
            nn.init.normal_(self.preconv.weight)
            self.preconv_scale = 1/math.sqrt(self.preconv.in_channels * self.preconv.kernel_size[0])
        self.equalized_lr = cfg.equalized_lr

        exponents = torch.arange(len(self.cfg.layer_specs) + 1) / (len(self.cfg.layer_specs) - cfg.num_critical)
        exponents = exponents.clamp_(0, 1)
        self.cutoffs = cfg.first_cutoff * (cfg.last_cutoff / cfg.first_cutoff) ** exponents 

        self.layers = []
        
        prev_seq_len = self.cfg.layer_specs[0][2]
        for i, (c_out, k, inner_seq_len) in enumerate(self.cfg.layer_specs):
            c_in = self.cfg.ff_channels if i == 0 else self.cfg.layer_specs[i-1][0]
            layer = RP_Layer(c_in, c_out, self.w_dim, k, self.cfg.lrelu_coeff, 
                             up_fc=self.cutoffs[i], down_fc=self.cutoffs[i+1],
                             up_factor=2 if prev_seq_len == inner_seq_len else 4, down_factor=2,
                             equalized_lr=cfg.equalized_lr
            )
            self.layers.append(layer)
            prev_seq_len = inner_seq_len
        self.layers = nn.ModuleList(self.layers)
        
        self.head = RP_Layer(c_out, self.cfg.c_dim, self.w_dim, 
                            self.cfg.head_spec[1], lrelu_coeff=1.0, # no activation.
                            normalize=False, up_factor=1, down_factor=1, equalized_lr=cfg.equalized_lr)

        print(f"[RP_W] Model initialized with {sum([p.numel() for p in self.parameters()]):,d} parameters")

    def forward(self, z: Tensor, seq_len: int, update_ema=False) -> Tensor:
        """ 
        `z` of shape (bs, z_dim)
        Returns `c` of shape (bs, seq_len, c_dim)
        """
        # z --> w
        w = self.W(z)
        # w --> seed fourier features
        #factor = int(seq_len*(self.first_seq_len/self.cfg.seq_len)) # e.g. 1/4
        seed = self.ff(w, self.first_seq_len)  # (bs, seq_len, ff_channels)
        # pass seed fourier features through pre-conv 
        c = seed.permute(0, 2, 1).contiguous()
        if self.equalized_lr:
            c = F.conv1d(c, self.preconv_scale*self.preconv.weight, self.preconv.bias, self.preconv.stride,
                         self.preconv.padding, self.preconv.dilation, self.preconv.groups)
        else:
            c = self.preconv(c) # (bs, ff_channels, seq_len)
        # Design layer which mixes w_vec with c and processes them that way.
        for layer in self.layers:
            c = layer(c, w, update_ema=update_ema)
        c = self.head(c, w, update_ema=update_ema)
        c = c.permute(0, 2, 1)
        return c
    
    def forward_w(self, w: Tensor, seq_len:int, update_ema=False) -> Tensor:
        # w --> seed fourier features
        factor = int(seq_len*(self.first_seq_len/self.cfg.seq_len)) # e.g. 1/4

        if w.dim() == 3:
            # per-layer inference, check that it is correct shape
            assert w.shape[1] == len(self.layers) + 2, "Per-layer w must be of shape (bs, n_layers+1, w_dim)!"
            seed = self.ff(w[:, 0], factor)  # (bs, seq_len, ff_channels)
        else:
            seed = self.ff(w, factor)  # (bs, seq_len, ff_channels)
        # pass seed fourier features through pre-conv 
        c = seed.permute(0, 2, 1).contiguous()
        if self.equalized_lr:
            c = F.conv1d(c, self.preconv_scale*self.preconv.weight, self.preconv.bias, self.preconv.stride,
                         self.preconv.padding, self.preconv.dilation, self.preconv.groups)
        else:
            c = self.preconv(c) # (bs, ff_channels, seq_len)
        # Design layer which mixes w_vec with c and processes them that way.

        for i, layer in enumerate(self.layers):
            if w.dim() == 3: c = layer(c, w[:, i+1], update_ema=update_ema)
            else: c = layer(c, w, update_ema=update_ema)

        if w.dim() == 3:
            c = self.head(c, w[:, -1], update_ema=update_ema)
        else:
            c = self.head(c, w, update_ema=update_ema)
        c = c.permute(0, 2, 1)
        return c

class RP_Layer(nn.Module):
    def __init__(self, c_in, c_out, w_dim, kernel_size=3, lrelu_coeff=0.1, ema_weight=0.001,
                normalize=True, up_factor=2, down_factor=2, up_fc=0.5, down_fc=0.5,
                equalized_lr=False) -> None:
        super().__init__()
        # at the moment no down-up sampling or other strange filtering.
        self.c_in = c_in
        self.c_out = c_out
        self.w_dim = w_dim
        self.ema_weight = ema_weight
        self.normalize = normalize
        self.equalized_lr = equalized_lr
        self.kernel_size = kernel_size

        self.affine = nn.Linear(w_dim, c_in)
        nn.init.ones_(self.affine.bias)

        self.conv = nn.Conv1d(c_in, c_out, kernel_size, stride=1, padding='same')
        nn.init.zeros_(self.conv.bias)
        self.relu = nn.LeakyReLU(lrelu_coeff) if lrelu_coeff < 1.0 else nn.Identity()
        self.register_buffer('norm_ema', torch.ones([]))

        if equalized_lr:
            nn.init.normal_(self.affine.weight, std=1)
            self.affine_w_scale = 1/math.sqrt(self.affine.in_features)
            nn.init.normal_(self.conv.weight, std=1)

        self.resampler = Resampler(up_factor=up_factor, down_factor=down_factor,
                                    up_fc=up_fc, down_fc=down_fc)

        
    def forward(self, x: Tensor, w: Tensor, update_ema) -> Tensor:
        """ `x` of shape (bs, c_in, seq_len), `w` of shape (bs, w_dim) """
        # 1. Normalize input
        if self.training and update_ema:
            # update emas:
            in_norm = x.detach().float().square().mean()
            self.norm_ema.copy_(torch.lerp(self.norm_ema, in_norm, self.ema_weight))
        elif self.training and not update_ema:
            logging.warn(("You are in training mode but not updating the emas."
                         " Please make sure you know what you are doing"))
        if self.equalized_lr == False:
            x = x * self.norm_ema.rsqrt()

        # 2. Pass w through affine layer
        if self.equalized_lr:
            conditioning = F.linear(w, self.affine_w_scale*self.affine.weight, self.affine.bias)
        else: conditioning = self.affine(w) # (bs, c_in)

        # 3. Perform modulated conv1d
        # 3.1 pre-normalize input weights. Note this isn't stated in StyleGAN2, 
        # but is done anyway cus magic
        if self.normalize:
            conv_w = self.conv.weight * self.conv.weight.square().mean([1, 2], keepdim=True).rsqrt() # (c_out, c_in, kernel_size)
            conv_s = conditioning * conditioning.square().mean().rsqrt() # (bs, c_in)
        else:
            # normalize conv_s according to StyleGAN3 toRGB layer:
            conv_w = self.conv.weight
            if self.equalized_lr:
                weight_gain = 1 / math.sqrt(self.c_in * self.kernel_size)
            else: 
                weight_gain = 1
            conv_s = conditioning * weight_gain

        # 3.2 Modulate weights
        conv_w = conv_w[None] * conv_s[:, None, :, None] # (1, c_out, c_in, k) * (bs, 1, c_in, 1) = (bs, c_out, c_in, k)
        # 3.3 Demodulate weights
        if self.normalize:
            dcoeffs = (conv_w.square().sum(dim=[2,3]) + 1e-8).rsqrt() # (bs, c_out)
            conv_w = conv_w * dcoeffs[..., None, None] # (bs, c_out, c_in, k)

        # 3.3.5 Apply input scaling (if equalized learning rates)
        if self.equalized_lr:
            conv_w = conv_w * self.norm_ema.rsqrt()

        # 3.4 Perform actual convolution
        # Magic way to do batched conv1d with a different kernel for each item in batch
        bs, c_out, c_in, k = conv_w.shape
        seq_len = x.shape[-1]
        w_ = conv_w.view(bs*c_out, c_in, k)
        x_ = x.view(1, bs*c_in, seq_len)
        o = F.conv1d(x_, w_, padding='same', groups=bs) 
        o = o.view(bs, c_out, seq_len)
        # 3.5 add in bias after 
        o = o + self.conv.bias[None, :, None] # add bias to each output channel

        # 4. 
        # 4.1 Upsample
        if self.resampler.up_factor > 1:
            o = self.resampler.up(o)
        # 4.2 Apply LeakyReLU
        o = self.relu(o)
        o = o * math.sqrt(2)
        # 4.3 Downsample
        if self.resampler.down_factor > 1:
            o = self.resampler.down(o)
        
        return o # (bs, c_out, seq_len)


class FourierFeature(nn.ModuleList):
    """ 1D Fourier features, excluding cosine features (only includes sine features). """
    def __init__(self, w_dim, channels, init_B_std=4, equalized_lr=False) -> None:
        """ sr=50 for hubert features, as they are every 20ms <-> 50Hz """
        super().__init__()
        self.w_dim = w_dim
        self.channels = channels
        # self.sr = sr # Hz
        B = torch.randn(channels)*init_B_std
        # phases = torch.rand(channels) - 0.5
        # Sort it purely for visualization purposes.
        B = B[B.abs().argsort(descending=False)]
        self.register_buffer('B', B)
        # self.register_buffer('phases', phases)
        logging.info(f"[Fourier Feature] B initialized with std: {init_B_std}")

        self.affine = nn.Linear(self.w_dim, 1)
        torch.nn.init.normal_(self.affine.weight, std=0.02)
        torch.nn.init.zeros_(self.affine.bias)

        self.mlp = nn.Linear(self.channels, self.channels, bias=False)
        nn.init.xavier_uniform_(self.mlp.weight)
        self.equalized_lr = equalized_lr
        if equalized_lr:
            nn.init.normal_(self.mlp.weight, std=1)
            self.mlp_w_scale = 1 / math.sqrt(self.mlp.in_features)
            self.affine_w_scale = 1 / math.sqrt(self.affine.in_features)
            nn.init.zeros_(self.affine.weight)
            nn.init.zeros_(self.affine.bias)


    def forward(self, w: Tensor, seq_len: int, sr: int = None) -> Tensor:
        """ Get Fourier features (bs, seq_len, channels) from a `w` vec (bs, w_dim) """
        if sr == None: sr = seq_len
        grid = torch.linspace(0, seq_len//sr, seq_len, dtype=w.dtype, device=w.device)
        if self.equalized_lr:
            phase = F.linear(w, self.affine_w_scale*self.affine.weight, self.affine.bias)
        else: phase = self.affine(w) # (bs, 1)
        # t_x = t[:, 0]
        # f_x = t[:, 1]
        # freqs = self.B[None] * f_x[:, None] # (bs, n_channels)
        # phases = self.phases[None] + self.B[None]*t_x[:, None] # (bs, channels)
        f = self.B[:, None]@grid[None] 
        f = f[None].repeat(w.shape[0], 1, 1) # ([bs, channels, seq_len])
        # gamma from the original Fourier Feature paper.
        # f = freqs[..., None]@grid[None, None] # (bs, channels, )
        # f = f + phases[..., None]
        gamma = torch.sin(2*math.pi*(f + phase[:, 0, None, None]))
        # gamma = torch.sin(2*math.pi*f)
        gamma = gamma.permute(0, 2, 1)

        # why divide by sqrt(channels)? Cus of equalized learning rate
        if self.equalized_lr:
            feats = F.linear(gamma, self.mlp_w_scale*self.mlp.weight)
        else:
            feats = self.mlp(gamma) 
        
        return feats # (bs, seq_len, channels)


class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size=4, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x: Tensor) -> Tensor:
        """ Adapted from StyleGAN2 official implementation. `x` of shape (bs, channels, seq_len) """
        N, C, H = x.shape
        G = min(self.group_size, N)
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x

class RP_ConvDiscriminator(nn.Module):

    def __init__(self, c_dim, d_dim, seq_len, lrelu_coeff, 
                        kernel_size=3, block_repeats=[3, 3, 3, 3], equalized_lr=False) -> None:
        super().__init__()
        self.c_dim = c_dim
        self.d_dim = d_dim
        self.seq_len = seq_len
        self.equalized_lr = equalized_lr
        if equalized_lr:
            print("[D] Using equalized lr")
        
        t_seq_lens = []
        rolling_seq_len = seq_len
        for i, b in enumerate(block_repeats):
            d = self.c_dim if i == 0 else d_dim
            if i == 0 and FIX_DISCRIMINATOR_FIRST_BLOCK:
                ext = [(rolling_seq_len, self.c_dim),] + [(rolling_seq_len, d_dim),]*(b-1)
                t_seq_lens.extend(ext)
            else:
                t_seq_lens.extend([(rolling_seq_len, d),]*b)
            # rolling_seq_len = rolling_seq_len//2
            rolling_seq_len = math.ceil(rolling_seq_len/2)

        # t_seq_lens = ([(self.seq_len, self.c_dim),]*3 +
        #             [(self.seq_len//2, d_dim),]*3 + 
        #             [(self.seq_len//4, d_dim),]*3 +
        #             [(self.seq_len//8, d_dim),]*3 + 
        # )
        # t_seq_lens = [2 ** i for i in range(int(math.log2(self.seq_len)), 2, -1)]
        #channels_dict = {res: min(32768 // res, 512) for res in t_seq_lens + [3]}
        
        # channels_dict[t_seq_lens[0]] = self.c_dim # override initial
        # logging.info(f"[RP_ConvD] layer sequence lengths:channels (base on {seq_len}): {channels_dict}")
        logging.info(f"[RP_ConvD] layer sequence lengths:channels (base on {seq_len}): {t_seq_lens}")
        logging.info(f"[RP_ConvD] sequence length at head: {t_seq_lens[-1][0]}")

        self.layers = []
        for i, (sl, in_ch) in enumerate(t_seq_lens[:-1]):
            #in_channels = channels_dict[res]
            in_channels = in_ch
            out_channels = t_seq_lens[i+1][1]
            # out_channels = channels_dict[res//2]
            in_seq_len = sl
            out_seq_len = t_seq_lens[i+1][0]
            block = ConvDBlock(in_channels, out_channels, lrelu_coeff, kernel_size=kernel_size, 
                                down_factor=math.ceil(in_seq_len/out_seq_len), equalized_lr=equalized_lr)
            self.layers.append(block)
            # for j in range(intermediate_layers):
                # another_block = ConvDBlock(out_channels, out_channels, lrelu_coeff, equalized_lr=equalized_lr)
                # self.layers.append(another_block)
            
        self.layers = nn.ModuleList(self.layers)
        self.head = ConvDHead(d_dim, head_seq_len=t_seq_lens[-1][0], kernel_size=kernel_size,
                                lrelu_coeff=lrelu_coeff, equalized_lr=equalized_lr) 

    def forward(self, c: Tensor) -> Tensor:
        """ `c` of shape (bs, seq_len, c_dim) """
        c = c.permute(0, 2, 1)
        for i, layer in enumerate(self.layers):
            c = layer(c)
        pred = self.head(c)
        return pred

class ConvDBlock(nn.Module):

    def __init__(self, in_channels, out_channels, lrelu_coeff, kernel_size=3, down_factor=2, clamp=256, equalized_lr=False) -> None:
        super().__init__()
        self.act = nn.LeakyReLU(lrelu_coeff)
        self.conv0 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding='same')
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=down_factor, padding=kernel_size//2 if down_factor > 1 else 'same')
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding='same', bias=False)
        nn.init.zeros_(self.conv0.bias)
        nn.init.zeros_(self.conv1.bias)

        self.clamp = clamp
        # Making 2x downsample filter:
        self.down_factor = down_factor
        self.resampler = Resampler(up_factor=1, down_factor=down_factor)
        if down_factor > 1:
            self.lpf = Resampler(1, 1)

        self.equalized_lr = equalized_lr
        if equalized_lr:
            nn.init.normal_(self.conv0.weight)
            nn.init.normal_(self.conv1.weight)
            nn.init.normal_(self.skip.weight)
            self.conv0_scale = 1/math.sqrt(in_channels * self.conv0.kernel_size[0])
            self.conv1_scale = 1/math.sqrt(in_channels * self.conv1.kernel_size[0])
            self.skip_scale = 1/math.sqrt(in_channels * self.skip.kernel_size[0])

    def forward(self, c):
        """ `c` of shape (bs, c_dim, seq_len) """
        # skip path:
        if self.down_factor > 1:
            s = self.resampler.down(c)
        else: s = c
        
        if self.equalized_lr:
            s = F.conv1d(s, self.skip_scale*self.skip.weight, None, 
                        self.skip.stride, self.skip.padding, self.skip.dilation, self.skip.groups)
            s = s * math.sqrt(2)
            c = F.conv1d(c, self.conv0_scale*self.conv0.weight, self.conv0.bias, 
                        self.conv0.stride, self.conv0.padding, self.conv0.dilation, self.conv0.groups)
            c = self.act(c.clamp_(-self.clamp, self.clamp))
            # why multiply by sqrt(2)? StyleGAN2 does it, and i think it has to do with keeping same norm
            # probably not a major influence.
            if self.down_factor > 1:
                c = self.lpf.lpf(c, self.lpf.down_lpf_filter)
            c = F.conv1d(c, self.conv1_scale*self.conv1.weight, self.conv1.bias, 
                        self.conv1.stride, self.conv1.padding, self.conv1.dilation, self.conv1.groups)
            c = c.clamp_(-self.clamp, self.clamp)
            c = c * math.sqrt(2)
        else: 
            s = self.skip(s) # (bs, out_channels, seq_len//2)
            s = s * math.sqrt(2)
            # main path:
            c = self.act(self.conv0(c).clamp_(-self.clamp, self.clamp))
            if self.down_factor > 1:
                c = self.lpf.lpf(c, self.lpf.down_lpf_filter)
            c = self.conv1(c).clamp_(-self.clamp, self.clamp)
            c = c * math.sqrt(2)
        # c = self.resampler.lpf(c, self.resampler.down_lpf_filter)
        c = self.act(c)
        s = s.add_(c)
        return s
         

class ConvDHead(nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        head_seq_len,                     # Resolution of this block.
        lrelu_coeff         = 0.1,
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        clamp               = 256,
        equalized_lr        = False,
        kernel_size         = 3
    ):
        super().__init__()
        self.in_channels = in_channels
        self.act = nn.LeakyReLU(lrelu_coeff)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels)
        self.conv = nn.Conv1d(in_channels + mbstd_num_channels, in_channels, kernel_size=kernel_size, padding='same')
        self.fc = nn.Linear(in_channels * head_seq_len, in_channels) # head_seq_len=4 typically
        self.out = nn.Linear(in_channels, 1)
        nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.fc.bias)
        nn.init.zeros_(self.conv.bias)
        self.equalized_lr = equalized_lr
        self.clamp = clamp
        if equalized_lr:
            nn.init.normal_(self.fc.weight)
            nn.init.normal_(self.out.weight)
            self.fc_w_scale = 1/math.sqrt(self.fc.in_features)
            self.out_w_scale = 1/math.sqrt(self.out.in_features)

            nn.init.normal_(self.conv.weight)
            self.conv_w_scale = 1 / math.sqrt(in_channels * self.conv.kernel_size[0])

    def forward(self, c: Tensor) -> Tensor:
        c = self.mbstd(c)
        if self.equalized_lr == False:
            c = self.act(self.conv(c).clamp_(-self.clamp, self.clamp)) # (bs, channels, seq_len=4)
            c = c.flatten(1) # (bs, channels*head_seq_len=4)
            c = self.act(self.fc(c))
            c = self.out(c)
        else:
            c = F.conv1d(c, self.conv_w_scale*self.conv.weight, self.conv.bias, 
                        self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
            c = self.act(c.clamp_(-self.clamp, self.clamp))*math.sqrt(2) # (bs, channels, seq_len=4)
            c = c.flatten(1)
            c = self.act(F.linear(c, self.fc_w_scale*self.fc.weight, self.fc.bias))*math.sqrt(2)
            c = F.linear(c, self.out_w_scale*self.out.weight, self.out.bias)
        return c


class Resampler(nn.Module):

    def __init__(self, up_factor:int=2, down_factor:int=2, M=8, beta=2.5, 
                 up_fc='critical', down_fc='critical') -> None:
        """ M must be a power of 2. """
        super().__init__()
        # Making 2x downsample filter:
        self.up_factor = up_factor
        self.down_factor = down_factor
        self.up_fc = up_fc
        self.down_fc = down_fc
        self.M = M
        self.beta = beta
        # Construct downsample LPF filter
        if down_fc == 'critical':
            cutoff = math.pi/down_factor # critical cutoff for 2x downsampling
            f0 = 1/down_factor
            self.down_fc = 0.5
        else:
            cutoff = down_fc*math.pi
            f0 = down_fc
        M_down = int(M*down_factor)
        n = torch.arange(-M_down//2, M_down//2 + 1).float()
        f = torch.sin(cutoff*n)/(math.pi*n + 1e-8)
        f[n.shape[0]//2] = f0
        window = torch.kaiser_window(M_down+1, False, beta)
        down_lpf_filter = window*f
        self.register_buffer('down_lpf_filter', down_lpf_filter)
        # Construct upsample LPF interpolation filter
        if up_fc == 'critical':
            cutoff = math.pi/up_factor # critical cutoff for 2x downsampling
            f0 = 1/up_factor
            self.up_fc = 0.5
        else:
            cutoff = up_fc*math.pi # Thus 
            f0 = up_fc
        M_up = int(M*up_factor)
        n = torch.arange(-M_up//2, M_up//2 + 1).float()
        f2 = torch.sin(cutoff*n)/(math.pi*n + 1e-8)
        f2[n.shape[0]//2] = f0
        window = torch.kaiser_window(M_up+1, False, beta)
        up_lpf_filter = window*f2
        self.register_buffer('up_lpf_filter', up_lpf_filter)

    def up(self, c: Tensor) -> Tensor:
        """ 
        Upsampling: inserts zeros and convolves with LPF (interpolation filter) 
        `c` of shape (bs, channels, seq_len)
        """
        bs, channels, seq_len = c.shape
        if self.up_factor != 1: 
            # Inserting zeros
            c_ = c[..., None] # (bs, channels, seq_len, 1)
            c_ = torch.nn.functional.pad(c_, [0, self.up_factor - 1])
            c_ = c_.reshape([bs, channels, seq_len * self.up_factor])
        else: c_ = c
        # Interpolation filter
        c_ = self.lpf(c_, self.up_lpf_filter)
        return c_

    def down(self, c: Tensor) -> Tensor:
        """ 
        Downsampling: apply LPF and only keep every `down_factor` input
        `c` of shape (bs, channels, seq_len)."""
        # LPF
        c = self.lpf(c, filter=self.down_lpf_filter)
        if self.down_factor == 1: return c
        # Downsample
        c = c[:, :, ::self.down_factor]
        return c


    def lpf(self, c: Tensor, filter: Tensor) -> Tensor:
        """ critical down_factor LPF of `c` of shape (bs, channels, seq_len) """
        n_channels = c.shape[1]
        f_ = filter.view(1, 1, -1)
        f_ = f_.expand(n_channels, 1, -1)
        s = F.conv1d(c, f_, stride=1, groups=n_channels, padding='same')
        return s

    def extra_repr(self):
        return (f"up_factor={self.up_factor:d}, down_factor={self.down_factor}, M={self.M}, beta={self.beta}, "
                f"up_fc={self.up_fc:3.2f}, down_fc={self.down_fc:3.2f}"
        )


# --------------------------------------------
# StyleGAN-3 type fourier feature


class SynthesisInput(nn.Module):
    """ Synthesis input module adapted from StyleGAN3 paper """
    def __init__(self, w_dim, channels, seq_len, equalized_lr):
        super().__init__()
        self.w_dim = w_dim
        self.channels = channels
        self.size = torch.tensor([seq_len, 1])
        self.sampling_rate = seq_len/2
        self.bandwidth = 2.0

        # Draw random frequencies from uniform 2D disc.
        freqs = torch.randn([self.channels, 2])
        radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
        freqs /= radii * radii.square().exp().pow(0.25)
        freqs *= self.bandwidth
        freqs = freqs[radii.abs().squeeze().argsort(descending=False)]
        phases = torch.rand([self.channels]) - 0.5

        # Setup parameters and buffers.
        self.affine = nn.Linear(w_dim, 4)
        torch.nn.init.zeros_(self.affine.weight)
        torch.nn.init.zeros_(self.affine.bias)
        self.affine.bias.data[0] = 1

        self.mlp = nn.Linear(self.channels, self.channels, bias=False)
        self.equalized_lr = equalized_lr
        if equalized_lr:
            nn.init.normal_(self.mlp.weight, std=1)
            self.mlp_w_scale = 1 / math.sqrt(self.mlp.in_features)
            self.affine_w_scale = 1 / math.sqrt(self.affine.in_features)

        self.register_buffer('transform', torch.eye(3, 3)) # User-specified inverse transform wrt. resulting image.
        self.register_buffer('freqs', freqs)
        self.register_buffer('phases', phases)

    def forward(self, w, dummy=None):
        # Introduce batch dimension.
        transforms = self.transform.unsqueeze(0) # [batch, row, col]
        freqs = self.freqs.unsqueeze(0) # [batch, channel, xy]
        phases = self.phases.unsqueeze(0) # [batch, channel]

        # Apply learned transformation.
        if self.equalized_lr:
            t = F.linear(w, self.affine_w_scale*self.affine.weight, self.affine.bias)
        else: t = self.affine(w) # t = (r_c, r_s, t_x, t_y)
        t = t / t[:, :2].norm(dim=1, keepdim=True) # t' = (r'_c, r'_s, t'_x, t'_y)
        m_r = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse rotation wrt. resulting image.
        m_r[:, 0, 0] = t[:, 0]  # r'_c
        m_r[:, 0, 1] = -t[:, 1] # r'_s
        m_r[:, 1, 0] = t[:, 1]  # r'_s
        m_r[:, 1, 1] = t[:, 0]  # r'_c
        m_t = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse translation wrt. resulting image.
        m_t[:, 0, 2] = -t[:, 2] # t'_x
        m_t[:, 1, 2] = -t[:, 3] # t'_y
        transforms = m_r @ m_t @ transforms # First rotate resulting image, then translate, and finally apply user-specified transform.

        # Transform frequencies.
        phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)
        freqs = freqs @ transforms[:, :2, :2]

        # Dampen out-of-band frequencies that may occur due to the user-specified transform.
        amplitudes = (1 - (freqs.norm(dim=2) - self.bandwidth) / (self.sampling_rate / 2 - self.bandwidth)).clamp(0, 1)
        # Construct sampling grid.
        theta = torch.eye(2, 3, device=w.device)
        theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
        theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate
        grids = torch.nn.functional.affine_grid(theta.unsqueeze(0), [1, 1, self.size[1], self.size[0]], align_corners=False)
        # grids is (1, height, width, 2)
        # Compute Fourier features.
        # freqs is (bs, channels, 2) --> (bs, 2, channels) -->  (1, height, width, 1, 2) @ (bs, 1, 1, 2, channels) 
        x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3) # [batch, height, width, 1, channel].squeeze(3)
        x = x + phases.unsqueeze(1).unsqueeze(2)
        x = torch.sin(x * (math.pi * 2))
        x = x * amplitudes.unsqueeze(1).unsqueeze(2)
        # Apply trainable mapping.
        if self.equalized_lr: 
            x = F.linear(x, self.mlp_w_scale*self.mlp.weight)
        else: 
            x = self.mlp(x)
        x = x.squeeze(1)
        # Ensure correct shape.
        # x = x.permute(0, 3, 1, 2) # [batch, channel, height, width]
        # x = x.squeeze(2) # (bs, channels, seq_len) 
        return x