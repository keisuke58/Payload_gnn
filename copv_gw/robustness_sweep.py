#!/usr/bin/env python3
"""COPV guided-wave damage-detection ROBUSTNESS: (1) across all 5 excitation freqs
@700 bar (healthy-control floor available), (2) across pressure 250-700 bar @180 kHz.
DI = 1 - corr(test, baseline@same-condition), per pitch-catch pair. [R] real BAM."""
import h5py, numpy as np, glob, os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = '/home/nishioka/GNN/external_datasets/bam_copv_guidedwave'
BASE = B + '/extracted_T25_baseline'; RDD = B + '/extracted_T25_rd'; IDD = B + '/extracted_T25_id'
REF700 = BASE + '/25-06-02_08-15-21_GW_Baseline_T25_700bar.h5'
CTRL700 = BASE + '/25-06-02_10-10-03_GW_Baseline_T25_700bar.h5'
W0 = 250
# 5 burst freqs -> first-rep raw index
FREQ_IDX = [(60, 0), (120, 3), (180, 6), (260, 9), (300, 12)]

def load(fn, ridx):
    h = h5py.File(fn, 'r'); s = h['Data/Raw_Data'][ridx, :, :].astype(float)[:, W0:]; h.close(); return s

def di_vec(ref, test):
    d = np.zeros(ref.shape[0])
    for i in range(ref.shape[0]):
        a = ref[i] - ref[i].mean(); b = test[i] - test[i].mean()
        d[i] = 1.0 - float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return d

def auroc(pos, neg):
    y = np.concatenate([np.ones(pos.size), np.zeros(neg.size)]); s = np.concatenate([pos, neg])
    order = np.argsort(s); r = np.empty_like(order, float); r[order] = np.arange(1, s.size + 1)
    return (r[y == 1].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)

def pressure_of(fn):
    m = re.search(r'_(\d+)bar', fn); return int(m.group(1)) if m else None
def byP(d):
    out = {}
    for fn in glob.glob(d + '/*bar.h5'):
        out.setdefault(pressure_of(fn), fn)
    return out

# ---------- (1) frequency robustness @700 bar ----------
print('=== frequency robustness @700 bar ===')
print('%5s | AUROC_rev AUROC_irr | meanDI_rev meanDI_irr | floor(ctrl)' % 'kHz')
freq_rows = []
for fk, ridx in FREQ_IDX:
    ref = load(REF700, ridx); ctrl = load(CTRL700, ridx)
    rd = load(glob.glob(RDD + '/*700bar.h5')[0], ridx); idf = load(glob.glob(IDD + '/*700bar.h5')[0], ridx)
    d_ctrl = di_vec(ref, ctrl); d_rd = di_vec(ref, rd); d_id = di_vec(ref, idf)
    a_rd = auroc(d_rd, d_ctrl); a_id = auroc(d_id, d_ctrl)
    freq_rows.append((fk, a_rd, a_id, d_rd.mean(), d_id.mean(), d_ctrl.mean()))
    print('%5d |   %.3f     %.3f  |   %.3f      %.3f   |  %.3f'
          % (fk, a_rd, a_id, d_rd.mean(), d_id.mean(), d_ctrl.mean()))

# ---------- (2) pressure robustness @180 kHz ----------
print('\n=== pressure robustness @180 kHz (ref=baseline@P) ===')
ridx = 6
bp = byP(BASE); rp = byP(RDD); ip = byP(IDD)
common = sorted(set(bp) & set(rp) & set(ip), reverse=True)
floor700 = freq_rows[2][5]  # 180 kHz ctrl mean DI @700
print('%6s | meanDI_rev meanDI_irr  (floor@700=%.3f)' % ('bar', floor700))
pres_rows = []
for P in common:
    ref = load(bp[P], ridx); rd = load(rp[P], ridx); idf = load(ip[P], ridx)
    m_rd = di_vec(ref, rd).mean(); m_id = di_vec(ref, idf).mean()
    pres_rows.append((P, m_rd, m_id))
    print('%6d |   %.3f      %.3f' % (P, m_rd, m_id))

# ---------- figure ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
fk = [r[0] for r in freq_rows]
ax[0].plot(fk, [r[1] for r in freq_rows], 'o-', color='#fdae61', label='reversible AUROC')
ax[0].plot(fk, [r[2] for r in freq_rows], 's-', color='#d7301f', label='irreversible AUROC')
ax[0].axhline(0.5, ls='--', c='gray', lw=0.8)
ax[0].set_xlabel('excitation frequency [kHz]'); ax[0].set_ylabel('detection AUROC')
ax[0].set_title('(1) frequency robustness @700 bar [R]'); ax[0].set_ylim(0.4, 1.02)
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

P = [r[0] for r in pres_rows]
ax[1].plot(P, [r[1] for r in pres_rows], 'o-', color='#fdae61', label='reversible mean DI')
ax[1].plot(P, [r[2] for r in pres_rows], 's-', color='#d7301f', label='irreversible mean DI')
ax[1].axhline(floor700, ls='--', c='#2c7fb8', lw=1, label='healthy floor @700')
ax[1].set_xlabel('pressure [bar]'); ax[1].set_ylabel('mean DI (1 - corr)')
ax[1].set_title('(2) pressure robustness @180 kHz [R]'); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
fig.suptitle('BAM COPV guided-wave damage detection robustness (freq + pressure)', y=1.02)
fig.tight_layout()
out = '/home/nishioka/Payload2026/copv_gw/copv_robustness'
fig.savefig(out + '.png', bbox_inches='tight', dpi=165); fig.savefig(out + '.pdf', bbox_inches='tight')
np.savez(out + '.npz', freq_rows=np.array(freq_rows), pres_rows=np.array(pres_rows))
print('\nwrote', out + '.png')
