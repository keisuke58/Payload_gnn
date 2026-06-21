#!/usr/bin/env python3
"""Real BAM COPV guided-wave DAMAGE DETECTION (healthy vs reversible vs irreversible),
all at 700 bar / T25 so the only variable is damage state. Per pitch-catch pair (600),
damage index vs a baseline reference; a 2nd baseline gives the healthy false-alarm floor.
Honest multi-metric: per-pair DI distributions + AUROC (healthy-control vs damaged).
[R] real BAM data throughout."""
import h5py, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = '/home/nishioka/GNN/external_datasets/bam_copv_guidedwave'
REF  = B + '/extracted_T25_baseline/25-06-02_08-15-21_GW_Baseline_T25_700bar.h5'
B2   = B + '/extracted_T25_baseline/25-06-02_10-10-03_GW_Baseline_T25_700bar.h5'  # healthy control
RD   = B + '/extracted_T25_rd/25-06-06_11-21-52_GW_RD_T25_700bar.h5'
ID   = B + '/extracted_T25_id/25-06-13_08-52-27_GW_ID_T25_700bar.h5'
RIDX = [6, 7, 8]           # 180 kHz, 3 reps (0-based)
W0 = 250                   # skip initial excitation-crosstalk window (samples)

def load_mean(fn):
    h = h5py.File(fn, 'r')
    raw = h['Data/Raw_Data']
    sig = np.mean([raw[r, :, :].astype(float) for r in RIDX], axis=0)  # (600,7552) avg of reps
    h.close()
    return sig[:, W0:]

ref = load_mean(REF)
def di_pairs(test):
    """per-pair damage indices vs ref: (1-corr) and normalized residual energy."""
    dcorr = np.zeros(ref.shape[0]); dres = np.zeros(ref.shape[0])
    for i in range(ref.shape[0]):
        a = ref[i] - ref[i].mean(); b = test[i] - test[i].mean()
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        rho = float((a @ b) / (na * nb + 1e-30))
        dcorr[i] = 1.0 - rho
        dres[i] = float(np.linalg.norm(test[i] - ref[i]) / (np.linalg.norm(ref[i]) + 1e-30))
    return dcorr, dres

states = {}
for name, fn in [('healthy-ctrl', B2), ('reversible', RD), ('irreversible', ID)]:
    dc, dr = di_pairs(load_mean(fn))
    states[name] = (dc, dr)
    print('%-13s  DI(1-corr): mean=%.4f med=%.4f p95=%.4f | DI(res): mean=%.3f'
          % (name, dc.mean(), np.median(dc), np.percentile(dc, 95), dr.mean()))

def auroc(pos, neg):
    from numpy import concatenate as cc
    y = cc([np.ones(pos.size), np.zeros(neg.size)]); s = cc([pos, neg])
    order = np.argsort(s); r = np.empty_like(order, float); r[order] = np.arange(1, s.size + 1)
    rp = r[y == 1].sum(); n1 = pos.size; n0 = neg.size
    return (rp - n1 * (n1 + 1) / 2.0) / (n1 * n0)

hc = states['healthy-ctrl'][0]
print('\n--- detection AUROC (per-pair DI 1-corr; damaged vs healthy-control) ---')
for name in ('reversible', 'irreversible'):
    a = auroc(states[name][0], hc)
    print('  %-13s AUROC=%.3f' % (name, a))

# --- figure: DI distributions ---
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
colors = {'healthy-ctrl': '#2c7fb8', 'reversible': '#fdae61', 'irreversible': '#d7301f'}
for name in ('healthy-ctrl', 'reversible', 'irreversible'):
    dc = states[name][0]
    ax[0].hist(dc, bins=40, alpha=0.55, color=colors[name], label='%s (mean %.3f)' % (name, dc.mean()))
ax[0].set_xlabel('per-pair DI = 1 - corr(test, baseline)'); ax[0].set_ylabel('# pitch-catch pairs')
ax[0].set_title('(a) damage index distributions @180 kHz [R]\nhealthy-control = false-alarm floor')
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

# sorted DI per pair (severity ranking)
for name in ('healthy-ctrl', 'reversible', 'irreversible'):
    ax[1].plot(np.sort(states[name][0])[::-1], color=colors[name], lw=1.1, label=name)
ax[1].set_xlabel('pitch-catch pair (sorted by DI)'); ax[1].set_ylabel('DI (1 - corr)')
ax[1].set_title('(b) sorted DI: damage lifts the whole curve\nabove the healthy floor [R]')
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
fig.suptitle('BAM COPV guided-wave damage detection (700 bar, T25, 180 kHz, 600 pairs)', y=1.02)
fig.tight_layout()
out = '/home/nishioka/Payload2026/copv_gw/copv_damage_detect'
fig.savefig(out + '.png', bbox_inches='tight', dpi=170); fig.savefig(out + '.pdf', bbox_inches='tight')
print('\nwrote', out + '.png')
