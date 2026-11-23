"""Per-FPN-level norm of the SSL projection-head mixer weights (head.mixer: Conv1d(L*D -> D, 1))."""
import sys, json, torch
out = {}
for name, path in [a.split('=', 1) for a in sys.argv[1:]]:
    ck = torch.load(path, map_location='cpu')
    res = {}
    for sd_key in ('state_dict', 'state_dict_ema'):
        sd = ck.get(sd_key)
        if sd is None:
            continue
        k = [k for k in sd if k.endswith('head.mixer.weight')][0]
        W = sd[k].squeeze(-1).float()            # (D, L*D)
        D = W.shape[0]; L = W.shape[1] // D
        norms = [W[:, l*D:(l+1)*D].norm().item() for l in range(L)]
        res[sd_key] = norms
    # fresh nn.Conv1d init for reference (same shape)
    W0 = torch.nn.Conv1d(L*D, D, 1).weight.squeeze(-1)
    res['init_reference'] = [W0[:, l*D:(l+1)*D].norm().item() for l in range(L)]
    out[name] = res
    print(name, {k: [round(x, 3) for x in v] for k, v in res.items()})
json.dump(out, open('figs/ssl_mixer_levels.json', 'w'), indent=1)
