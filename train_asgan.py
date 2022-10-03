import argparse
import logging
import os
import random
import time
from itertools import chain
from pathlib import Path
import gc, copy

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from fastprogress import master_bar, progress_bar
from omegaconf import MISSING, OmegaConf, open_dict
from torch.cuda.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from density.dataset import HubertFeatureDataset, RP_Collate
from density.metrics import ApproxKL, closest_dist_pair, perc_closest, fad, make_grad_norm_fig, plot_melspecs
from density.models import RP_W, RP_ConvDiscriminator
from density.losses import (logistic_d_loss, logistic_g_loss, dist_heuristic_loss, r1_reg)
from density.augment import Augmenter, melspec2hubert

from density.config import *


def train(rank, cfg: TrainConfig):
    if cfg.distributed.n_gpus_per_node > 1:
        init_process_group(backend=cfg.distributed.dist_backend, init_method=cfg.distributed.dist_url,
                           world_size=cfg.distributed.n_nodes*cfg.distributed.n_gpus_per_node, rank=rank)

    device = torch.device(f'cuda:{rank:d}')

    if cfg.model == 'rp_w':
        g = RP_W(cfg.rp_w_cfg)
        g_ema = copy.deepcopy(g).eval()
        d = RP_ConvDiscriminator(cfg.c_dim, cfg.rp_w_cfg.D_head_dim, cfg.rp_w_cfg.seq_len,
                                 cfg.rp_w_cfg.lrelu_coeff, cfg.rp_w_cfg.D_kernel_size, cfg.rp_w_cfg.D_block_repeats,
                                 cfg.rp_w_cfg.equalized_lr)

        data_collater = RP_Collate(cfg.c_dim, cfg.rp_w_cfg.seq_len, cfg.data_type)
        model_cfg = cfg.rp_w_cfg
    else: raise NotImplementedError

    g = g.to(device)
    d = d.to(device)    
    logging.info(f"Initialized rank {rank}")
    
    if rank == 0:
        logging.getLogger().setLevel(logging.INFO)
        logging.info(f"Generator initialized as:\n {g}")
        logging.info(f"Discriminator initialized as:\n {d}")
        os.makedirs(cfg.checkpoint_path, exist_ok=True)
        logging.info(f"checkpoints directory : {cfg.checkpoint_path}")
        print(f"G has {sum([p.numel() for p in g.parameters()]):,d} parameters.")
        print(f"D has {sum([p.numel() for p in d.parameters()]):,d} parameters.")

    steps = 0
    if cfg.resume_checkpoint != '' and os.path.isfile(cfg.resume_checkpoint):
        state_dict = torch.load(cfg.resume_checkpoint, map_location=device)
        g.load_state_dict(state_dict['g_state_dict'])
        if g_ema is not None:
            g_ema.load_state_dict(state_dict['g_ema_state_dict'])
        d.load_state_dict(state_dict['d_state_dict'])
        steps = state_dict['steps'] + 1
        last_epoch = state_dict['epoch']
        print(f"Checkpoint loaded from {cfg.resume_checkpoint}. Resuming training from {steps} steps at epoch {last_epoch}")
    else:
        state_dict = None
        last_epoch = -1

    if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1:
        if rank == 0: logging.info("Multi-gpu detected")
        g = DDP(g, device_ids=[rank]).to(device)
        d = DDP(d, device_ids=[rank]).to(device)

    # Set up data and optimizers

    base_params = {'params': chain(g.ff.parameters(), g.preconv.parameters(), g.layers.parameters(), g.head.parameters())}
    g_optim = torch.optim.Adam(
                [
                    base_params,
                    {'params': g.W.parameters(), 'lr': cfg.lr*model_cfg.w_lr_mult if model_cfg.equalized_lr == False else cfg.lr}
                ], 
                cfg.lr, betas=cfg.betas, weight_decay=0)
    d_optim = torch.optim.Adam(d.parameters(), cfg.lr*cfg.d_lr_mult, betas=cfg.betas, weight_decay=0)
    if state_dict is not None:
        g_optim.load_state_dict(state_dict['g_optim_state_dict'])
        d_optim.load_state_dict(state_dict['d_optim_state_dict'])
        logging.warning("Not updating learning rates with current config. Using previous learning rates.")
    
    # Gather all files
    all_files = list(Path(cfg.train_root).rglob('**/*.pt'))
    if len(all_files) == 0:
        all_files = list(Path(cfg.train_root).rglob('**/*.wav'))
        all_files += list(Path(cfg.train_root).rglob('**/*.mp3'))
        all_files += list(Path(cfg.train_root).rglob('**/*.flac'))
    all_files = sorted(all_files)

    if cfg.use_sc09_splits:
        print("Using original SC09 train/valid/test split")
        train_csv = pd.read_csv(cfg.sc09_train_csv)
        valid_csv = pd.read_csv(cfg.sc09_valid_csv)
        train_files = list(train_csv.path)
        valid_files = list(valid_csv.path)
        if cfg.data_type == 'hubert_L6':
            # swap paths
            tr = Path(cfg.train_root)
            train_files = [tr/(Path(t).relative_to(Path(t).parents[1])).with_suffix('.pt') for t in train_files]
            valid_files = [tr/(Path(t).relative_to(Path(t).parents[1])).with_suffix('.pt') for t in valid_files]
    else:
        random.shuffle(all_files)
        train_files = all_files[:-cfg.n_valid]
        valid_files = all_files[-cfg.n_valid:]

    if rank == 0:
        logging.info(f"Found {len(all_files)} files, split into {len(train_files)}-{len(valid_files)} train-valid split.")

    train_ds = HubertFeatureDataset(train_files, cfg.z_dim, cfg.z_mean, cfg.z_std, cfg.preload, cfg.data_type)

    train_sampler = DistributedSampler(train_ds) if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1 else None
    train_dl = DataLoader(train_ds, num_workers=cfg.num_workers, shuffle=False if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1 else True,
                              sampler=train_sampler,
                              batch_size=cfg.batch_size,
                              pin_memory=False,
                              drop_last=True,
                              collate_fn=data_collater)

    if rank == 0:

        valid_ds = HubertFeatureDataset(valid_files, cfg.z_dim, cfg.z_mean, cfg.z_std, cfg.preload, cfg.data_type)
        if cfg.data_type == 'melspec':
            hifigan = torch.hub.load("bshall/hifigan:main", "hifigan", map_location='cpu').cpu().eval()
            hubert = torch.hub.load('RF5/simple-asgan', 'hubert_base', device='cpu')
            # get hifigan stats 
            valid_ds.prep_fad_metrics(hifigan=hifigan, hubert=hubert)
        elif cfg.data_type == 'wav':
            hubert = torch.hub.load('RF5/simple-asgan', 'hubert_base', device='cpu')
            valid_ds.prep_fad_metrics(hifigan=None, hubert=hubert)
        else:
            valid_ds.prep_fad_metrics()
        valid_dl = DataLoader(valid_ds, num_workers=cfg.num_workers, shuffle=False,
                                       sampler=None,
                                       batch_size=cfg.batch_size,
                                       pin_memory=False,
                                       drop_last=True,
                                       collate_fn=data_collater)

        sw = SummaryWriter(os.path.join(cfg.checkpoint_path, 'logs'))
        sw.add_text('config', '```\n' + OmegaConf.to_yaml(cfg) + '\n```', global_step=steps)

    if cfg.fp16: 
        scaler_g = GradScaler(init_scale=512)
        scaler_d = GradScaler(init_scale=512)
        if state_dict is not None and state_dict['g_scaler_state_dict'] is not None:
            scaler_g.load_state_dict(state_dict['g_scaler_state_dict'])
            scaler_d.load_state_dict(state_dict['d_scaler_state_dict'])

    g.train()
    d.train()
    if state_dict is not None:
        del state_dict['g_state_dict']
        del state_dict['d_state_dict']
        del state_dict['g_scaler_state_dict']
        del state_dict['d_scaler_state_dict']
        del state_dict['g_optim_state_dict']
        del state_dict['d_optim_state_dict']
    torch.cuda.empty_cache()
    
    if rank == 0: 
        mb = master_bar(range(max(0, last_epoch), cfg.n_epochs))
        smooth_loss_g = None
        smooth_loss_d = None
        smooth_r1 = 0
        r1_loss = 0
        loss_d = torch.tensor(float('nan'))
        mean_term = torch.tensor(float(0))
        var_term = torch.tensor(float(0))
        dist_loss = torch.tensor(0.0)
        d_gnorm = 0
        g_gnorm = 0
        loss_g = torch.tensor(0)
        kl_metric = ApproxKL()
        best_fad = float('inf')
    else: mb = range(max(0, last_epoch), cfg.n_epochs)

    d_update_perc = 1.0
    augmenter = Augmenter(p=cfg.aug_init_p, noise_std=0.05*(4 if cfg.data_type == 'melspec' else 1))

    # ----------------------------------------------
    # Main training loop
    # ----------------------------------------------
    update = 'D'
    inner_cnt = 0

    for epoch in mb:
        if rank == 0:
            start = time.time()
            mb.write("Epoch: {}".format(epoch+1))

        if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1:
            train_sampler.set_epoch(epoch)

        if rank == 0: pb = progress_bar(enumerate(train_dl), total=len(train_dl), parent=mb)
        else: pb = enumerate(train_dl)        

        for i, batch in pb:
            if rank == 0: start_b = time.time()
            c_real, lengths, _, z = batch
            # c_real (bs, seq_len, c_dim)
            # lengths (bs)
            # next_feats (bs, c_dim)
            # z (bs, z_dim)
            c_real = c_real.to(device, non_blocking=True)
            z = z.to(device, non_blocking=True)

            # ----------------------
            # Update discriminator

            # use D augmenter to see if we will skip D update
            skip_d = augmenter.skip_d_update()
            if update == 'D' and skip_d == True and not (steps % 16 == 0):
                update = 'G'


            if update == 'D':
                d_optim.zero_grad()

                with autocast(enabled=cfg.fp16):
                    with torch.no_grad():
                        c_fake = g(z, model_cfg.seq_len, update_ema=True)
                        c_fake = c_fake.detach()

                    c_real, c_fake = augmenter(c_real, c_fake)
                    fake_outs = d(c_fake)
                    
                    if steps > 0 and steps % model_cfg.apply_r1_every == 0:
                        # apply R1 regularization:
                        c_real_ = c_real.requires_grad_(True)
                        real_outs = d(c_real_)
                        r1_loss = r1_reg(real_outs, c_real_, model_cfg.r1_gamma)
                    else:
                        # skip R1 regularization
                        real_outs = d(c_real)
                        r1_loss = 0

                    lsgan_loss_d = logistic_d_loss(real_outs, fake_outs)
                    loss_d = lsgan_loss_d + r1_loss

                if (i % cfg.update_ratio == 0 or lsgan_loss_d > cfg.d_loss_update_max_threshold) and lsgan_loss_d > cfg.d_loss_update_min_threshold:
                    if cfg.fp16: 
                        scaler_d.scale(loss_d).backward()
                        scaler_d.unscale_(d_optim)
                        d_gnorm = torch.nn.utils.clip_grad.clip_grad_norm_(d.parameters(), cfg.grad_clip)
                        scaler_d.step(d_optim)
                        scaler_d.update()
                    else: 
                        loss_d.backward()
                        d_gnorm = torch.nn.utils.clip_grad.clip_grad_norm_(d.parameters(), cfg.grad_clip)
                        d_optim.step()
                    # did update
                    d_update_perc = d_update_perc + 0.05*(1 - d_update_perc)
                else:
                    # did not update
                    d_update_perc = d_update_perc + 0.05*(0 - d_update_perc)
                
                # Update ADA with d_real_outs
                if cfg.data_type != 'wav':
                    augmenter.accumulate(real_outs)
                # Switching accumulate
                inner_cnt += 1
                if inner_cnt >= cfg.update_ratio:
                    inner_cnt = 0
                    update = 'G'

            elif update == 'G':
                # ----------------------
                # Update generator
                g_optim.zero_grad()
                d_update_perc = d_update_perc + 0.05*(0 - d_update_perc)

                with autocast(enabled=cfg.fp16):
                    c_fake = g(z, model_cfg.seq_len, update_ema=True)

                    c_real, c_fake = augmenter(c_real, c_fake)
                    fake_outs_with_grad = d(c_fake)
                    # loss_g = g_loss(fake_outs_with_grad)
                    loss_g = logistic_g_loss(fake_outs_with_grad)

                    mean_term = F.mse_loss(c_fake.mean(dim=[0, 1]), c_real.mean(dim=[0, 1]))
                    var_term = F.mse_loss(c_fake.std(dim=[0, 1]), c_real.std(dim=[0, 1]))
                    
                    # distance losses for metric use only
                    dist_loss = dist_heuristic_loss(c_real.reshape(-1, cfg.c_dim), c_fake.reshape(-1, cfg.c_dim), n=75)
                    if dist_loss.numel() > 0: dist_loss = dist_loss.mean() # we do not need to pull closer than 0.4
                    else: dist_loss = 0

                if cfg.fp16:
                    scaler_g.scale(loss_g).backward()
                    scaler_g.unscale_(g_optim)
                    g_gnorm = torch.nn.utils.clip_grad.clip_grad_norm_(g.parameters(), cfg.grad_clip)
                    scaler_g.step(g_optim)
                    scaler_g.update()
                else:
                    loss_g.backward()
                    g_gnorm = torch.nn.utils.clip_grad.clip_grad_norm_(g.parameters(), cfg.grad_clip)
                    g_optim.step()

                # Switching accumulate
                inner_cnt += 1
                if inner_cnt >= cfg.update_ratio:
                    inner_cnt = 0
                    update = 'D'
                
            if rank == 0 and (steps > 5 if last_epoch == -1 else steps > state_dict['steps']+5):
                if smooth_loss_g is None: smooth_loss_g = float(loss_g.item())
                else: smooth_loss_g = smooth_loss_g + 0.1*(float(loss_g.item()) - smooth_loss_g)
                if smooth_loss_d is None: smooth_loss_d = float(loss_d.item())
                else: smooth_loss_d = smooth_loss_d + 0.1*(float(loss_d.item()) - smooth_loss_d)
                if r1_loss > 1e-8:
                    smooth_r1 = smooth_r1 + 0.2*(float(r1_loss.item()) - smooth_r1)
                # STDOUT logging
                if steps % cfg.stdout_interval == 0:
                    mb.write('Steps: {:,d}, g_loss: {:4.3f}, d_loss: {:4.3f}, dmetric: {:4.3f} , s/b: {:4.3f}, mem: {:5.2f}GB'. \
                            format(steps, loss_g, loss_d, 0.0, time.time() - start_b, torch.cuda.max_memory_reserved()/1e9))
                mb.child.comment = 'Steps: {:,d}, g_loss: {:4.3f}, d_loss: {:4.3f}, s/batch: {:4.3f}'. \
                        format(steps, loss_g, loss_d, time.time() - start_b)
                    
                # checkpointing
                if steps % cfg.checkpoint_interval == 0 and steps != 0:
                    with torch.no_grad():
                        if g_ema is not None:
                            # Update exponential moving average weights
                            for p_ema, p in zip(g_ema.parameters(), g.parameters()):
                                p_ema.copy_(p_ema.lerp(p.detach().to(p_ema.device), 
                                    model_cfg.g_ema_weight if cfg.model == 'rp_w' else cfg.rp_w_tfm_cfg.g_ema_weight))
                            for b_ema, b in zip(g_ema.buffers(), g.buffers()):
                                b_ema.copy_(b.detach().to(b_ema.device))

                    # Saving checkpoint
                    checkpoint_path = f"{cfg.checkpoint_path}/ckpt_{steps:08d}.pt"
                    torch.save({
                        'g_state_dict': (g.module if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1 else g).state_dict(),
                        'd_state_dict': (d.module if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1 else d).state_dict(),
                        'd_optim_state_dict': d_optim.state_dict(),
                        'g_optim_state_dict': g_optim.state_dict(),
                        'g_ema_state_dict': g_ema.state_dict() if g_ema is not None else None,
                        'd_scaler_state_dict': (scaler_d.state_dict() if cfg.fp16 else None),
                        'g_scaler_state_dict': (scaler_g.state_dict() if cfg.fp16 else None),
                        'steps': steps,
                        'epoch': epoch,
                        'cfg_yaml': OmegaConf.to_yaml(cfg)
                    }, checkpoint_path)
                    logging.info(f"Saved checkpoint to {checkpoint_path}")

                # Tensorboard summary logging
                if steps % cfg.summary_interval == 0:
                    sw.add_scalar("memory/sec_per_batch", time.time() - start_b, steps)
                    sw.add_scalar("training/g_loss_smooth", smooth_loss_g, steps)
                    sw.add_scalar("training/d_loss_smooth", smooth_loss_d, steps)
                    sw.add_scalar("opt/g_lr", float(g_optim.param_groups[0]['lr']), steps)
                    sw.add_scalar("opt/g_W_lr", float(g_optim.param_groups[1]['lr']), steps)
                    sw.add_scalar("opt/d_lr", float(d_optim.param_groups[0]['lr']), steps)
                    sw.add_scalar('opt/g_grad_norm', float(g_gnorm), steps)
                    sw.add_scalar('opt/d_grad_norm', float(d_gnorm), steps)
                    sw.add_scalar('training/mean_drift', float(mean_term), steps)
                    sw.add_scalar('training/var_drift', float(var_term), steps)
                    sw.add_scalar('training/dist_loss', float(dist_loss), steps)
                    sw.add_scalar('training/r1_loss', float(smooth_r1), steps)
                    sw.add_scalar('training/total_d_loss', float(loss_d), steps)
                    sw.add_scalar('opt/d_update_perc', d_update_perc, steps)
                    sw.add_scalar('training/D(G(z))', fake_outs.detach().mean(), steps)
                    sw.add_scalar('training/D(c)', real_outs.detach().mean(), steps)
                    sw.add_scalar('data/ada_p', augmenter.p, steps)
                    sw.add_scalar('data/r_t', augmenter.rt, steps)
                if steps % cfg.grad_summary_interval == 0:
                    # Summarize gradients
                    grad_fig = make_grad_norm_fig(
                        g.module if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1 else g,
                        d.module if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1 else d
                    )
                    sw.add_figure("opt/grad_norms", grad_fig, steps)

                # Validation
                if steps % cfg.validation_interval == 0:  # and steps != 0:
                    g.eval()
                    d.eval()
                    kl_metric.clear()
                    torch.cuda.empty_cache()
                    g_val_loss_tot = 0
                    d_val_loss_tot = 0
                    c_fakes = []
                    d_of_fake = []
                    d_of_real = []
                    with torch.no_grad():
                        for j, batch in progress_bar(enumerate(valid_dl), total=len(valid_dl), parent=mb):
                            c_real, lengths, _, z = batch
                            c_real = c_real.to(device, non_blocking=True)
                            z = z.to(device, non_blocking=True)
                            
                            c_fake = g(z, model_cfg.seq_len)
                            fake_outs = d(c_fake.detach())
                            real_outs = d(c_real)
                            d_of_real.append(real_outs.cpu())
                            d_of_fake.append(fake_outs.cpu())
                            # a = d_loss(real_outs, fake_outs)
                            a = logistic_d_loss(real_outs, fake_outs)
                            d_val_loss_tot += a
                            # g_val_loss_tot += g_loss(fake_outs)
                            g_val_loss_tot += logistic_g_loss(fake_outs)

                            kl_metric.accumulate(c_real.reshape(-1, cfg.c_dim), c_fake.reshape(-1, cfg.c_dim))

                            if j < 1:
                                n_collect = min(1000, c_real.shape[0]*c_real.shape[1]) # c_real.shape[0]
                                c_r_ = c_real.reshape(-1, cfg.c_dim)
                                c_f_ = c_fake.reshape(-1, cfg.c_dim)
                                collect_inds = torch.randperm(c_r_.shape[0])[:n_collect]
                                lblls = ['real' for _ in range(n_collect)] + ['fake' for _ in range(n_collect)]
                                if steps % int(cfg.validation_interval*10) == 0:
                                    sw.add_embedding(torch.cat((c_r_[collect_inds], c_f_[collect_inds]), dim=0), lblls, global_step=steps)
                            
                                m1, m2 = closest_dist_pair(c_r_, c_f_)
                                gc.collect()
                                torch.cuda.empty_cache()
                                sw.add_scalars('validation/min_cosim_dists', {'c_fake_nearest_dist': m1.mean(),
                                                                              'c_real_nearest_dist': m2.mean()}, steps)
                                sw.add_histogram('validation/c_fake_nearest_dists', m1, steps)
                                sw.add_histogram('validation/c_real_nearest_dists', m2, steps)
                                fake_closest_to_real_perc, real_closest_to_fake_perc = perc_closest(c_r_, c_f_, n=500)
                                sw.add_scalar('validation/fake_closest_to_real_perc', fake_closest_to_real_perc, steps)
                                sw.add_scalar('validation/real_closest_to_fake_perc', real_closest_to_fake_perc, steps)

                                if cfg.data_type == 'melspec':
                                    fig = plot_melspecs(c_fake)
                                    sw.add_figure("mel/generated_mels", fig, steps)

                            if cfg.data_type in ['melspec',]: c_fakes.append(c_fake.cpu())
                            else: c_fakes.append(c_fake.reshape(-1, cfg.c_dim).cpu())

                        c_fakes = torch.cat([c.float() for c in c_fakes], dim=0)
    
                        if cfg.data_type == 'melspec':
                            # convert c_fakes --> audio --> hubert feats
                            c_fakes = melspec2hubert(c_fakes, hubert, hifigan, mb=mb)
                            c_fakes = torch.flatten(c_fakes, 0, 1)

                        fad_measure = fad(c_fakes, valid_ds.mean, valid_ds.cov)
                        del c_fakes
                        d_val_loss = d_val_loss_tot / (j+1)
                        g_val_loss = g_val_loss_tot / (j+1)
                        d_of_real = torch.stack(d_of_real)
                        d_of_fake = torch.stack(d_of_fake)
                        sw.add_histogram('validation/D(G(z))', d_of_fake, steps)
                        sw.add_histogram('validation/D(c)', d_of_real, steps)
                        del d_of_fake
                        del d_of_real
                        sw.add_scalar("validation/g_loss", g_val_loss, steps)
                        sw.add_scalar("validation/d_loss", d_val_loss, steps)
                        sw.add_scalar("validation/fad", float(fad_measure), steps)
                        sw.add_scalar("validation/approx_kl_div", kl_metric.metric(), steps)
                        sw.add_figure("validation/c_val_hist", kl_metric.make_plot(), steps)
                        mb.write((f"validation run complete at {steps:,d} steps. g loss: {g_val_loss:5.4f} | d loss: {d_val_loss:5.4f} "
                                  f" | fad: {fad_measure:5.4f}"))
                        
                        # Keep track of best checkpoint.
                        if fad_measure < best_fad:
                            best_fad = fad_measure
                            checkpoint_path = f"{cfg.checkpoint_path}/ckpt_best_fad.pt"
                            torch.save({
                                'g_state_dict': (g.module if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1 else g).state_dict(),
                                'd_state_dict': (d.module if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1 else d).state_dict(),
                                'd_optim_state_dict': d_optim.state_dict(),
                                'g_optim_state_dict': g_optim.state_dict(),
                                # 'scheduler_state_dict': scheduler.state_dict(),
                                'd_scaler_state_dict': (scaler_d.state_dict() if cfg.fp16 else None),
                                'g_scaler_state_dict': (scaler_g.state_dict() if cfg.fp16 else None),
                                'steps': steps,
                                'epoch': epoch,
                                'cfg_yaml': OmegaConf.to_yaml(cfg)
                            }, checkpoint_path)
                            logging.info(f"New best checkpoint {checkpoint_path} with FAD {fad_measure:5.4f}")


                    g.train()
                    d.train()
                    sw.add_scalar("memory/max_allocated_gb", torch.cuda.max_memory_allocated()/1e9, steps)
                    sw.add_scalar("memory/max_reserved_gb", torch.cuda.max_memory_reserved()/1e9, steps)
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()
                    torch.cuda.empty_cache()
                    gc.collect()


            steps += 1
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(usage='\n' + '-'*10 + ' Default config ' + '-'*10 + '\n' + 
                            str(OmegaConf.to_yaml(OmegaConf.structured(TrainConfig))))
    a = parser.parse_known_args()
    override_cfg = OmegaConf.from_cli()
    base_cfg = OmegaConf.structured(TrainConfig)
    cfg: TrainConfig = OmegaConf.merge(base_cfg, override_cfg)
    logging.info(f"Running with config:\n {OmegaConf.to_yaml(cfg)}")

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        if cfg.distributed.n_gpus_per_node > torch.cuda.device_count():
            raise AssertionError((f" Specified n_gpus_per_node ({cfg.distributed.n_gpus_per_node})"
                                    f" must be less than or equal to cuda device count ({torch.cuda.device_count()}) "))
        with open_dict(cfg):
            cfg.batch_size_per_gpu = int(cfg.batch_size / cfg.distributed.n_gpus_per_node)
        if cfg.batch_size % cfg.distributed.n_gpus_per_node != 0:
            logging.warn(("Batch size does not evenly divide among GPUs in a node. "
                            "Likely unbalanced loads will occur."))
        logging.info(f'Batch size per GPU : {cfg.batch_size_per_gpu}')

    if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1:
       mp.spawn(train, nprocs=cfg.distributed.n_gpus_per_node, args=(cfg,))
    else:
       train(0, cfg)


if __name__ == '__main__':
    main()
