# python imports
import argparse
import os
import time
import datetime
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
from torch.utils.tensorboard import SummaryWriter

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.modeling.selfsup import (make_ssl_loss, make_ssl_optimizer,
                                   ema_momentum_schedule)
from libs.utils import (save_checkpoint, make_scheduler,
                        fix_random_seed, ModelEma, AverageMeter)


################################################################################
def build_views(video_list, view_idx, div_factor, max_seq_len, device, dtype=None):
    """
    Slice one temporal view per video and pad the batch to a multiple of
    div_factor (required by the multi-scale transformer / local attention).

    Returns (x (B, C, P), mask (B, 1, P) bool, view_start (B,) float).
    """
    B = len(video_list)
    C = video_list[0]['feats'].size(0)
    if dtype is None:
        dtype = video_list[0]['feats'].dtype

    starts = []
    lens = []
    for v in video_list:
        s, e = v['crop_box'][view_idx]
        starts.append(int(s))
        lens.append(int(e - s))

    P = max(lens)
    P = min(max_seq_len, ((P + div_factor - 1) // div_factor) * div_factor)
    P = max(P, div_factor)

    x = torch.zeros(B, C, P, dtype=dtype, device=device)
    mask = torch.zeros(B, 1, P, dtype=torch.bool, device=device)
    for i, v in enumerate(video_list):
        L = lens[i]
        if L <= 0:
            continue
        s, e = v['crop_box'][view_idx]
        x[i, :, :L].copy_(v['feats'][:, s:e].to(device).to(dtype))
        mask[i, 0, :L] = True

    view_start = torch.tensor(starts, dtype=torch.float32, device=device)
    return x, mask, view_start


def build_roi(video_list, device, dtype=torch.float32):
    """
    Shared temporal ROI = intersection of the two crops, in the original
    feature-grid coordinates.
    """
    B = len(video_list)
    rs = torch.zeros(B, dtype=dtype, device=device)
    re = torch.zeros(B, dtype=dtype, device=device)
    for i, v in enumerate(video_list):
        (s1, e1), (s2, e2) = v['crop_box']
        rs[i] = max(s1, s2)
        re[i] = min(e1, e2)
    return torch.stack([rs, re], dim=1)


def calc_ema_momentum(global_step, total_steps, ema_cfg):
    return ema_momentum_schedule(
        global_step, total_steps,
        start_momentum=ema_cfg.get('start_momentum', 0.9995),
        end_momentum=ema_cfg.get('end_momentum', 1.0),
    )


def train_one_epoch_ssl(
    train_loader,
    model,           # student (already DataParallel)
    teacher,         # ModelEma(student)
    loss_fn,
    optimizer,
    scheduler,
    curr_epoch,
    ema_cfg,
    ema_total_steps,
    clip_grad_l2norm,
    tb_writer,
    print_freq,
    div_factor,
    max_seq_len,
    master_device,
):
    batch_time = AverageMeter()
    losses_tracker = {}
    num_iters = len(train_loader)
    model.train()

    print("\n[Train]: Epoch {:d} started".format(curr_epoch))
    start = time.time()
    for iter_idx, video_list in enumerate(train_loader, 0):
        optimizer.zero_grad(set_to_none=True)

        # view 1 -> student, view 2 -> EMA teacher
        x1, m1, st1 = build_views(video_list, 0, div_factor, max_seq_len, master_device)
        x2, m2, st2 = build_views(video_list, 1, div_factor, max_seq_len, master_device)
        roi = build_roi(video_list, master_device)

        f_s, amask_s = model(x1, m1, st1, roi)
        with torch.no_grad():
            f_t, amask_t = teacher.module(x2, m2, st2, roi)

        losses = loss_fn(f_s, f_t, amask_s & amask_t)
        losses['final_loss'].backward()

        if clip_grad_l2norm > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)

        optimizer.step()
        scheduler.step()

        # EMA teacher with the cosine momentum schedule
        momentum = calc_ema_momentum(
            curr_epoch * num_iters + iter_idx, ema_total_steps, ema_cfg)
        teacher.decay = momentum
        teacher.update(model)

        # printing / logging
        if (iter_idx != 0) and (iter_idx % print_freq) == 0:
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            for key, value in losses.items():
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                losses_tracker[key].update(value.item())

            if tb_writer is not None:
                lr = scheduler.get_last_lr()[0]
                global_step = curr_epoch * num_iters + iter_idx
                for key, value in losses_tracker.items():
                    if key == 'final_loss':
                        tb_writer.add_scalar('train/final_loss', value.val, global_step)
                    else:
                        tb_writer.add_scalar('train/' + key, value.val, global_step)
                tb_writer.add_scalar('train/learning_rate', lr, global_step)
                tb_writer.add_scalar('train/ema_momentum', momentum, global_step)

            block1 = 'Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                curr_epoch, iter_idx, num_iters)
            block2 = 'Time {:.2f} ({:.2f})'.format(
                batch_time.val, batch_time.avg)
            loss_blocks = ''
            for key, value in losses_tracker.items():
                loss_blocks += '\t{:s} {:.3f} ({:.3f})'.format(
                    key, value.val, value.avg)
            print(block1 + ' ' + block2 + loss_blocks)
    return


def validate_ssl(
    val_loader,
    model,
    teacher,
    loss_fn,
    div_factor,
    max_seq_len,
    master_device,
):
    """Compute the SSL loss / top-1 neighbor agreement on the val split."""
    model.eval()
    losses_tracker = {}
    cnt = 0
    with torch.no_grad():
        for video_list in val_loader:
            x1, m1, st1 = build_views(video_list, 0, div_factor, max_seq_len, master_device)
            x2, m2, st2 = build_views(video_list, 1, div_factor, max_seq_len, master_device)
            roi = build_roi(video_list, master_device)
            f_s, amask_s = model.module(x1, m1, st1, roi)
            f_t, amask_t = teacher.module(x2, m2, st2, roi)
            losses = loss_fn(f_s, f_t, amask_s & amask_t)
            for key, value in losses.items():
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                losses_tracker[key].update(value.item())
            cnt += 1
    msg = '[Val]: '
    for key, value in losses_tracker.items():
        msg += '{:s} {:.4f} | '.format(key, value.avg)
    print(msg)
    return losses_tracker


def _master_device(cfg):
    dev = cfg['devices'][0]
    if isinstance(dev, str):
        return torch.device(dev)
    return torch.device('cuda:%d' % dev)


def load_init_encoder(model, ckpt_path, device):
    """
    Warm-start the SSL encoder from a supervised ActionFormer checkpoint
    (or a previous SSL run). Only backbone / neck weights are copied
    (keys starting with 'module.' / 'backbone.' / 'neck.'); the extra
    'head.' parameters keep their random init.
    """
    checkpoint = torch.load(ckpt_path, map_location=device)

    state = checkpoint['state_dict'] if 'state_dict' in checkpoint \
        else checkpoint
    # strip the optional DataParallel prefix
    state = {k[len('module.'):] if k.startswith('module.') else k: v
             for k, v in state.items()}

    # keep only the shared encoder parameters (drop the head, and drop
    # any stale supervised heads: cls/reg/center pre-convs)
    enc = {}
    for k, v in state.items():
        if k.startswith('head.'):
            continue
        if any(k.startswith(p) for p in ('cls_head', 'reg_head', 'center_head')):
            continue
        enc[k] = v

    missing, unexpected = model.load_state_dict(enc, strict=False)
    print(">> init-encoder: loaded encoder weights from {:s} "
          "({:d} matched, {:d} missing, {:d} unexpected)".format(
              ckpt_path, len(enc) - len(unexpected),
              len(missing), len(unexpected)))
    return model


################################################################################
def main(args):
    """main function that handles self-supervised training"""

    """1. setup parameters / folders"""
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)

    # self-supervised method (flag overrides config)
    ssl_cfg = cfg.get('ssl', {}) or {}
    ssl_method = args.ssl_method if args.ssl_method is not None else \
        ssl_cfg.get('method', 'neco')
    print("SSL method: {:s}".format(ssl_method))

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create dataset / dataloader"""
    # self-supervised dataset: no GT needed; adds ssl-specific crop params
    dataset_kwargs = dict(cfg['dataset'])
    dataset_kwargs['crop_scale'] = ssl_cfg.get('crop_scale', [0.4, 1.0])
    dataset_kwargs['min_overlap'] = ssl_cfg.get('min_overlap', 0.2)
    dataset_kwargs['min_crop_len'] = ssl_cfg.get('min_crop_len', 128)

    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **dataset_kwargs)
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **dataset_kwargs)

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])
    val_loader = make_data_loader(
        val_dataset, False, None, cfg['loader']['batch_size'],
        cfg['loader']['num_workers'])

    # NeCo needs >= 2 videos per mini-batch (reference pool)
    assert cfg['loader']['batch_size'] >= 2, \
        "NeCo requires batch_size >= 2 (reference pool from other videos)"

    """3. create model, teacher (EMA), loss, optimizer, and scheduler"""

    # warm-start the encoder from a supervised (or previous SSL) checkpoint
    args.init_encoder = getattr(args, 'init_encoder', '')
    model = make_meta_arch('PtTransformerSSL', **cfg['model'], ssl_cfg=ssl_cfg)
    if args.init_encoder:
        model = load_init_encoder(
            model, args.init_encoder, _master_device(cfg))

    # build the optimizer BEFORE wrapping with DataParallel so that
    # head-vs-encoder parameter groups can be separated cleanly
    head_lr_scale = ssl_cfg.get('head_lr_scale', 10.0)
    optimizer = make_ssl_optimizer(model, cfg['opt'], head_lr_scale=head_lr_scale)

    # teacher / student share the architecture; teacher = EMA of student
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    # scheduler
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # loss (add-on friendly flag; only 'neco' is implemented)
    loss_fn = make_ssl_loss(ssl_method, ssl_cfg)

    """4. Resume from model / Misc"""
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(
                args.resume, map_location=lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d})".format(
                args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """5. training / validation loop"""
    print("\nStart SSL post-training (method = {:s}) ...".format(ssl_method))

    master_device = _master_device(cfg)
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )
    total_steps = max_epochs * num_iters_per_epoch
    ema_cfg = ssl_cfg.get('ema', {}) or {}

    for epoch in range(args.start_epoch, max_epochs):
        train_one_epoch_ssl(
            train_loader,
            model,
            model_ema,
            loss_fn,
            optimizer,
            scheduler,
            epoch,
            ema_cfg,
            ema_total_steps=total_steps,
            clip_grad_l2norm=cfg['train_cfg'].get('clip_grad_l2norm', -1),
            tb_writer=tb_writer,
            print_freq=args.print_freq,
            div_factor=model.module.max_div_factor,
            max_seq_len=model.module.max_seq_len,
            master_device=master_device,
        )

        # lightweight validation (NeCo loss / agreement on the val split)
        if (epoch + 1) % args.val_every == 0 or (epoch + 1) == max_epochs:
            val_stats = validate_ssl(
                val_loader, model, model_ema, loss_fn,
                div_factor=model.module.max_div_factor,
                max_seq_len=model.module.max_seq_len,
                master_device=master_device,
            )
            for key, value in val_stats.items():
                tb_writer.add_scalar('validation/' + key, value.avg, epoch + 1)

        # save ckpt once in a while
        if (
            ((epoch + 1) == max_epochs) or
            ((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0))
        ):
            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch + 1)
            )

    # wrap up
    tb_writer.close()
    print("All done!")

    # fine-tune hint: the encoder weights now live in the _ema checkpoint
    print(">> The SSL encoder is stored in state_dict_ema of the last ckpt.")
    print(">> For supervised fine-tuning, load it into LocPointTransformer")
    print(">>   (keys match; just drop the extra 'head.' parameters).")
    return


################################################################################
if __name__ == '__main__':
    """Entry Point"""
    parser = argparse.ArgumentParser(
        description='Self-supervised (NeCo-style) post-training of ActionFormer')
    parser.add_argument('config', metavar='DIR', help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=20, type=int,
                        help='print frequency (default: 20 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('-v', '--val-every', default=1, type=int,
                        help='validate every N epochs (default: 1)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    parser.add_argument('--init-encoder', default='', type=str, metavar='PATH',
                        help='warm-start the SSL encoder from a supervised '
                             'ActionFormer ckpt (default: from random init)')
    parser.add_argument('--ssl-method', default=None,
                        choices=['neco', 'dino', 'byol', 'masked'],
                        help='self-supervised method; overrides config '
                             '(only neco implemented, others reserved)')
    args = parser.parse_args()
    main(args)