# python imports
import argparse
import glob
import os
import re
import time
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed


################################################################################
def _find_supervised_ckpt(ckpt_dir, config_name):
    """
    Auto-discover the supervised checkpoint for the given config.

    Scans  ckpt_dir  for sub-folders that start with the config name
    (e.g.  epic_vjepa2_verb_reproduce). Returns the path of
    model_best.pth.tar  if it exists, otherwise the latest
    epoch_NNN.pth.tar  in the newest matching folder. Returns None if
    nothing is found.
    """
    candidates = glob.glob(os.path.join(ckpt_dir, config_name + '*'))
    candidates = sorted(
        [d for d in candidates if os.path.isdir(d)],
        key=os.path.getmtime, reverse=True,
    )
    if not candidates:
        return None

    for folder in candidates:
        best = os.path.join(folder, 'model_best.pth.tar')
        if os.path.isfile(best):
            return best
        epochs = sorted(glob.glob(os.path.join(folder, 'epoch_*.pth.tar')),
                        key=lambda f: int(re.search(r'epoch_(\d+)', f).group(1)))
        if epochs:
            return epochs[-1]
    return None


def _strip_module(state):
    """Remove the optional DataParallel 'module.' prefix from state keys."""
    return {k[len('module.'):] if k.startswith('module.') else k: v
            for k, v in state.items()}


def _load_state(ckpt_path, use_ema=True):
    """Load a checkpoint and return its (stripped) state dict,
    preferring the EMA weights."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if use_ema and 'state_dict_ema' in ckpt:
        state = ckpt['state_dict_ema']
        print("   (using EMA weights)")
    elif 'state_dict' in ckpt:
        state = ckpt['state_dict']
        print("   (using student weights)")
    else:
        state = ckpt
        print("   (raw state dict, no EMA/student keys)")
    return _strip_module(state)


################################################################################
def main(args):
    """0. load config"""
    # sanity check: this is the SUPERVISED config (dataset / model /
    # test_cfg of the downstream task), not the ssl_*.yaml
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['val_split']) > 0, "Test set must be specified!"

    # resolve the SSL checkpoint (file, or folder + epoch / latest)
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ssl_ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ssl_ckpt_file = os.path.join(
                args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch)
            )
        else:
            ckpt_file_list = sorted(
                glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            ssl_ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ssl_ckpt_file)

    # resolve the supervised checkpoint that provides the cls/reg heads
    # (auto-discover from ckpt/ if --head-from is not given)
    if len(args.head_from) > 0:
        sup_ckpt_file = args.head_from
        assert os.path.isfile(sup_ckpt_file), \
            "--head-from checkpoint does not exist!"
    else:
        cfg_name = os.path.basename(args.config).replace('.yaml', '')
        sup_ckpt_file = _find_supervised_ckpt(args.ckpt_dir, cfg_name)
    assert sup_ckpt_file is not None, (
        "Could not find a supervised checkpoint for the cls/reg heads. "
        "Pass it explicitly with --head-from PATH."
    )

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    pprint(cfg)

    """1. fix all randomness"""
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model"""
    # the SUPERVISED ActionFormer (LocPointTransformer): backbone / neck
    # will be overwritten by the SSL encoder, cls_head / reg_head come
    # from the supervised checkpoint
    model = make_meta_arch(cfg['model_name'], **cfg['model'])

    """4. merge weights: SSL encoder + supervised heads"""
    print("=> supervised checkpoint (heads): {:s}".format(sup_ckpt_file))
    sup_state = _load_state(sup_ckpt_file, use_ema=True)

    print("=> SSL checkpoint (encoder): {:s}".format(ssl_ckpt_file))
    ssl_state = _load_state(ssl_ckpt_file, use_ema=not args.no_ema)

    # encoder-only keys from the SSL run (drop the SSL projection head)
    enc = {k: v for k, v in ssl_state.items()
           if k.startswith('backbone.') or k.startswith('neck.')}
    assert len(enc) > 0, \
        "No backbone./neck. keys found in the SSL checkpoint!"

    # start from the full supervised state, overwrite the encoder
    merged = dict(sup_state)
    merged.update(enc)

    missing, unexpected = model.load_state_dict(merged, strict=False)
    assert len(missing) == 0, \
        "missing keys after merge: {}".format(missing)
    assert len(unexpected) == 0, \
        "unexpected keys after merge: {}".format(unexpected)
    print("=> merged model: {:d} encoder keys from SSL, {:d} head keys "
          "from supervised checkpoint".format(
              len(enc), len(merged) - len(enc)))

    del sup_state, ssl_state, merged, enc

    model = nn.DataParallel(model, device_ids=cfg['devices'])

    # set up evaluator
    val_db_vars = val_dataset.get_attributes()
    det_eval = ANETdetection(
        val_dataset.json_file,
        val_dataset.split[0],
        tiou_thresholds = val_db_vars['tiou_thresholds']
    )

    """5. Test the model"""
    # optionally only dump the detections (e.g., for the test set)
    output_file = None
    if args.saveonly:
        output_file = ssl_ckpt_file.replace('.pth.tar', '') + '_dets.json'

    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    mAP = valid_one_epoch(
        val_loader,
        model,
        -1,
        evaluator=det_eval,
        output_file=output_file,
        ext_score_file=cfg['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=args.print_freq
    )
    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return


################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Evaluate a post-trained (SSL) ActionFormer: '
                  'SSL encoder + supervised cls/reg heads')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to the SUPERVISED config file '
                             '(e.g. configs/epic_vjepa2_verb.yaml)')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to the SSL checkpoint (file or folder)')
    parser.add_argument('-epoch', type=int, default=-1,
                        help='checkpoint epoch (default: latest)')
    parser.add_argument('--head-from', default='', type=str, metavar='PATH',
                        help='supervised checkpoint providing the cls/reg '
                             'heads (default: auto-discover from --ckpt-dir)')
    parser.add_argument('--ckpt-dir', default='ckpt', type=str,
                        help='root ckpt folder for auto-discovery of the '
                             'supervised checkpoint (default: ckpt)')
    parser.add_argument('--no-ema', action='store_true',
                        help='use the SSL student weights instead of EMA')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the outputs without evaluation '
                             '(e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    args = parser.parse_args()
    main(args)
