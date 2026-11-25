"""
Relative L2 distance between the backbone (or neck) weights of two checkpoints:
    ||W_b - W_a||_2 / ||W_a||_2
over all floating-point tensors under the given prefix, using each checkpoint's EMA
state ('state_dict_ema', i.e. the SSL teacher / the evaluated supervised model) when
present. Used e.g. to measure how far P3 stage B moves the encoder away from the SSL
teacher it starts from.

usage: python tools/weight_distance.py name=<ckpt_a>=<ckpt_b> [...] [--prefix backbone]
       (ckpt_a is the reference / starting point)
"""
import argparse
import json

import torch


def ema_state(path):
    ck = torch.load(path, map_location='cpu')
    sd = ck.get('state_dict_ema', ck.get('state_dict', ck))
    return {k.replace('module.', '', 1): v for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pairs', nargs='+', help='name=ckpt_a=ckpt_b (a = reference)')
    ap.add_argument('--prefix', default='backbone')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    out = {}
    for pair in args.pairs:
        name, a_path, b_path = pair.split('=')
        a, b = ema_state(a_path), ema_state(b_path)
        ks = sorted(k for k in a if k.startswith(args.prefix) and k in b
                    and a[k].is_floating_point() and a[k].shape == b[k].shape)
        va = torch.cat([a[k].flatten().float() for k in ks])
        vb = torch.cat([b[k].flatten().float() for k in ks])
        out[name] = float((vb - va).norm() / va.norm())
        print('%-24s %s rel. distance %.4f  (%d tensors)' % (name, args.prefix, out[name], len(ks)), flush=True)
    if args.out:
        json.dump(out, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
