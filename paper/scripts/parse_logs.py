"""Parse ALICE job logs (*.out) into compact JSON for the paper figures.

usage: python parse_logs.py <dir with *.out and data/epic_kitchens/annotations> <out dir>
Writes curves.json (per-epoch validation mAP / NeCo val metrics) and
durations.json (EPIC-100 training-video durations in seconds).
"""
import json, os, re, sys

src, dst = sys.argv[1], sys.argv[2]

SUP = {  # label -> log file (supervised; per-epoch validation avg mAP)
    'sf_base_s1': 'train_verb_seed_5094858.out',
    'sf_base_s2': 'train_verb_seed_5094861.out',
    'sf_p3_s0': 'train_verb_from_ssl_5077230.out',
    'sf_p3_s1': 'train_verb_from_ssl_seed_5094860.out',
    'sf_p3_s2': 'train_verb_from_ssl_seed_5094863.out',
    'sf_p3_rate': 'train_verb_from_ssl_rate_5094853.out',
    'vm_base': 'train_verb_videomae_5093406.out',
    'vm_p3_temporal': 'train_verb_videomae_from_ssl_5094855.out',
    'vm_p3_spatial': 'train_verb_videomae_from_ssl_spatial_5175414.out',
    'vm_p3_spatiotemporal': 'train_verb_videomae_from_ssl_spatial_temporal_5176611.out',
    'vm_p3_spatial_v2': 'train_verb_videomae_from_ssl_spatial_v2_5318166.out',
    'vj_base': 'train_verb_4621944.out',
    'vj_p3': 'train_verb_vjepa2_from_ssl_5093405.out',
    'ft_ctrl': 'ft_joint_ctrl_5093321.out',
    'ft_ssl_warm': 'ft_joint_ssl_warm_5093320.out',
    # corrected SSL pipeline (teacher references, EMA 0.99, EMA warm start), 2026-11-22
    'sf_p3f_s0': 'verb_from_ssl_fixed_5702534.out',
    'sf_p3f_s1': 'verb_from_ssl_fixed_5703154.out',
    'sf_p3f_s2': 'verb_from_ssl_fixed_5704521.out',
    'sf_p3f_lowlr': 'verb_from_ssl_fixed_5702538.out',
    'vm_p3f_temporal': 'verb_from_ssl_fixed_5704522.out',
    'vm_p3f_spatial': 'verb_from_ssl_fixed_5705830.out',
}
SSL = {  # label -> log file (SSL; per-epoch val NeCo loss / top-1 agreement)
    'sf_cold': 'ssl_neco_slowfast_4796515.out',
    'sf_cold_s1': 'ssl_neco_seed_5094859.out',
    'sf_cold_s2': 'ssl_neco_seed_5094862.out',
    'sf_rate': 'ssl_neco_rate_5094852.out',
    'sf_warm': 'ssl_neco_slowfast_warm_4988444.out',
    'vm_cold': 'ssl_neco_videomae_cold_5094854.out',
    'vm_warm': 'ssl_neco_videomae_warm_5094856.out',
    'vm_spatial': 'ssl_neco_videomae_spatial_5175368.out',
    'vm_spatiotemporal': 'ssl_neco_videomae_spatial_temporal_5176610.out',
    'vm_spatial_v2': 'ssl_neco_videomae_spatial_v2_5318111.out',
    'vj_cold': 'ssl_neco_vjepa2_cold_5093340.out',
    'vj_warm': 'ssl_neco_vjepa2_4917434.out',
    'sf_f_s0': 'ssl_fixed_5702531.out',
    'sf_f_s1': 'ssl_fixed_5702532.out',
    'sf_f_s2': 'ssl_fixed_5703155.out',
    'sf_f_lowlr': 'ssl_fixed_5702537.out',
    'sf_f_warm': 'ssl_fixed_5702539.out',
    'vm_f_temporal': 'ssl_fixed_5703157.out',
    'vm_f_spatial': 'ssl_fixed_5705829.out',
}

out = {'supervised': {}, 'ssl': {}}
def read(f):
    path = os.path.join(src, f)
    return open(path).read() if os.path.exists(path) else ''


for k, f in SUP.items():
    txt = read(f)
    out['supervised'][k] = [float(x) for x in re.findall(r'Average mAP: ([\d.]+)', txt)]
for k, f in SSL.items():
    txt = read(f)
    v = re.findall(r'\[Val\]: neco_loss ([\d.]+) \| neco_top1_agreement ([\d.]+)', txt)
    out['ssl'][k] = {'loss': [float(a) for a, _ in v], 'agree': [float(b) for _, b in v]}
json.dump(out, open(os.path.join(dst, 'curves.json'), 'w'), indent=1)

db = json.load(open(os.path.join(src, 'data/epic_kitchens/annotations/epic_kitchens_100_verb.json')))['database']
json.dump({k: v['duration'] for k, v in db.items() if v['subset'] == 'training'},
          open(os.path.join(dst, 'durations.json'), 'w'))
for k, v in out['supervised'].items():
    print(k, len(v), v[-1] if v else None)
for k, v in out['ssl'].items():
    print(k, len(v['loss']), v['loss'][:1], v['loss'][-1:], v['agree'][-1:])
# first-epoch running-average supervised loss (head-start probe)
ep1 = {}
for k, f in SUP.items():
    m = re.findall(r'Epoch: \[000\]\[\d+/\d+\]\s+Time [\d.]+ \([\d.]+\)\s+Loss [\d.]+ \(([\d.]+)\)', read(f))
    if m:
        ep1[k] = float(m[-1])
out['epoch1_avg_loss'] = ep1
json.dump(out, open(os.path.join(dst, 'curves.json'), 'w'), indent=1)
print('epoch-1 avg loss', ep1)
