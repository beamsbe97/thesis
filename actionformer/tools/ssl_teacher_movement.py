"""Relative L2 change from the seeded random init, for BOTH the student (state_dict)
and the EMA teacher (state_dict_ema, the weights stage B actually loads)."""
import json, sys, torch
sys.path.insert(0, '.')
from libs.core import load_config
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed

RUNS = [
    ('supervised SlowFast', 'ckpt/epic_slowfast_verb_reproduce/epoch_021.pth.tar', 'configs/epic_slowfast_verb.yaml', None),
    ('SSL SlowFast (temporal)', 'ckpt_ssl/ssl_epic_slowfast_verb_neco/epoch_012.pth.tar', 'configs/ssl_epic_slowfast_verb.yaml', None),
    ('SSL SlowFast seed 1', 'ckpt_ssl/ssl_epic_slowfast_verb_neco_seed1/epoch_012.pth.tar', 'configs/ssl_epic_slowfast_verb.yaml', 1),
    ('SSL SlowFast rate-aug', 'ckpt_ssl/ssl_epic_slowfast_verb_rate_neco/epoch_012.pth.tar', 'configs/ssl_epic_slowfast_verb_rate.yaml', None),
    ('SSL VideoMAE (temporal)', 'ckpt_ssl/ssl_epic_videomae_verb_neco_cold/epoch_012.pth.tar', 'configs/ssl_epic_videomae_verb.yaml', None),
    ('SSL VideoMAE spatial v1', 'ckpt_ssl/ssl_epic_videomae_verb_spatial_neco_spatial/epoch_012.pth.tar', 'configs/ssl_epic_videomae_verb_spatial.yaml', None),
    ('SSL VideoMAE spatial v2', 'ckpt_ssl/ssl_epic_videomae_verb_spatial_v2_neco_spatial_v2/epoch_020.pth.tar', 'configs/ssl_epic_videomae_verb_spatial_v2.yaml', None),
]

def strip(sd):
    return {k.replace('module.', '', 1): v for k, v in sd.items()}

# optional CLI override: name=ckpt=config[=seed] ...
if len(sys.argv) > 1:
    RUNS = []
    for arg in sys.argv[1:]:
        parts = arg.split('=')
        RUNS.append((parts[0], parts[1], parts[2], int(parts[3]) if len(parts) > 3 else None))

out = {}
for name, ckpt, cfgp, seed in RUNS:
    cfg = load_config(cfgp)
    fix_random_seed(seed if seed is not None else cfg['init_rand_seed'], include_cuda=False)
    init = {k: v.detach().clone() for k, v in make_meta_arch(cfg['model_name'], **cfg['model']).state_dict().items()}
    ck = torch.load(ckpt, map_location='cpu')
    res = {}
    for which in ('state_dict', 'state_dict_ema'):
        sd = strip(ck[which])
        r = {}
        for p in ('backbone', 'neck', 'head'):
            ks = [k for k in sd if k.startswith(p) and k in init and sd[k].shape == init[k].shape and sd[k].is_floating_point()]
            if not ks:
                continue
            a = torch.cat([sd[k].flatten().float() for k in ks]); b = torch.cat([init[k].flatten().float() for k in ks])
            r[p] = float((a - b).norm() / b.norm())
        res[which] = r
    out[name] = res
    print(name, json.dumps(res))
if len(sys.argv) == 1:
    json.dump(out, open('figs/teacher_student_delta.json', 'w'), indent=1)
