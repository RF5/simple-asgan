from dataclasses import dataclass, field
from typing import List, Tuple, Union

from omegaconf import MISSING, OmegaConf, open_dict
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from density.models import RP_W

def fix(blah): return field(default_factory=lambda: blah)


@dataclass
class DistributedConfig:
    dist_backend: str = 'nccl'
    dist_url: str = "tcp://localhost:54321"
    n_nodes: int = 1
    n_gpus_per_node: int = 1

   
@dataclass
class RP_W_Config:
    z_dim: int = 512
    c_dim: int = 768 # 128 for mel-spectrograms, 768 for hubert
    w_lr_mult: float = 0.01 # as per stylegan3 code.
    w_dim: int = 512
    w_layers: int = 2
    ff_channels: int = 512
    init_B_std: float = 0.1
    # ff_sr: int = 50

    seq_len: int = 48 # 96 for mel-spectrograms, 48 for hubert variant. 
    apply_r1_every: int = 8
    r1_gamma: float = 0.05
    
    first_cutoff: float = 0.125
    last_cutoff: float = 0.45 # Nyquist
    num_critical: int = 2

    # Generator network params
    # list of (channels, kernel size) for each layer
    ## Layer specs for mel-spectrograms
    # layer_specs: List = fix(
        # [(1024, 5, seq_len/8)] * 5 +
        # [(512, 5, seq_len/4)] * 4 +
        # [(256, 5, seq_len/2)] * 3 + 
        # [(128, 5, seq_len)] * 2
    # )
    ## Layer specs for hubert embeddings
    layer_specs: List = fix(
        [(1024, 3, seq_len/8)] * 5 +
        [(768, 3, seq_len/4)] * 4 +
        [(512, 3, seq_len/2)] * 3 + 
        [(512, 3, seq_len)] * 2
    )

    head_spec: Tuple[int, int] = fix((c_dim, 1))
    lrelu_coeff: float = 0.1

    # Discriminator network params
    # D_layers: int = 3
    # D_biridr: bool = False
    # D_hidden_dim: int = 512
    D_head_dim: int = 512
    D_kernel_size: int = 3
    # The default for below is [3, 3, 3, 3]
    D_block_repeats: List[int] = fix([3, 3, 3, 3]) 

    # equalized learning rates:
    equalized_lr: bool = True
    g_ema_weight: float = 0.1
    use_sg3_ff: bool = False

@dataclass
class TrainConfig:
    # Distributed settings
    distributed: DistributedConfig = DistributedConfig()
    # Model settings
    rv: bool = True
    c_dim: int = 768

    # latent settings
    z_dim: int = 512
    z_mean: float = 0.0
    z_std: float = 1.0

    device: str = 'cuda'
    preload: bool = False
    seed: int = 1775
    
    batch_size: int = 24
    num_workers: int = 8
    fp16: bool = True
    n_epochs: int = 50
    summary_interval: int = 50
    grad_summary_interval: int = summary_interval*10
    checkpoint_interval: int = 5000
    stdout_interval: int = 100
    validation_interval: int = 2500

    update_ratio: int = 1 # update D every `update_ratio` updates to G
    # a D loss above which we will force the D update through,
    # regardless of the current setting in `update_ratio`
    d_loss_update_max_threshold: float = 9999 #0.2
    # a D loss below which we will force the D update to be skipped,
    # regardless of the current setting in `update_ratio`
    d_loss_update_min_threshold: float = -9999
    
    # Learning settings
    # start_lr: float = 1e-6
    # max_lr: float = 5e-5
    # end_lr: float = 3e-7
    lr: float = 6e-5 # stylegan3 uses 2e-3
    # warmup_pct: float = 0.15
    betas: Tuple[float, float] = (0.8, 0.99)
    grad_clip: float = 1.0

    # Data settings
    checkpoint_path: str = MISSING
    train_root: str = MISSING
    n_valid: int = MISSING
    resume_checkpoint: str = ''
    # augmentation params
    aug_init_p: float = 0.05
    adapt_d_lr: bool = False
    d_lr_mult: float = 1

    rp_w_cfg: RP_W_Config = RP_W_Config
    model: str = 'rv_w' # either rv_w or rp_w
    data_type: str = 'hubert_L6'
    
    use_sc09_splits: bool = False
    sc09_valid_csv: str = ''
    sc09_train_csv: str = ''


def flatten_cfg(cfg: Union[DictConfig, ListConfig]) -> dict:
    """ 
    Recursively flattens a config into a flat dictionary compatible with 
    tensorboard's `add_hparams` function.
    """
    out_dict = {}
    if type(cfg) == ListConfig:
        cfg = DictConfig({f"[{i}]": v for i, v in enumerate(cfg)})

    for key in cfg:
        if type(getattr(cfg, key)) in (int, str, bool, float):
            out_dict[key] = getattr(cfg, key)
        elif type(getattr(cfg, key)) in [DictConfig, ListConfig]:
            out_dict = out_dict | {f"{key}{'.' if type(getattr(cfg, key)) == DictConfig else ''}{k}": v for k, v in flatten_cfg(getattr(cfg, key)).items()}
        else: raise AssertionError
    return out_dict
