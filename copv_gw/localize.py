#!/usr/bin/env python3
"""COPV guided-wave damage LOCALIZATION (real BAM, irreversible @180 kHz).
(A) per-PZT damage involvement on the 5x5 grid -> robust, needs only grid topology.
(B) RAPID elliptical tomography -> damage-probability image in ASSUMED uniform grid
    coordinates [U] (true PZT spacing is in the scanned PDF). Healthy-control shown
    as the null map. [R] real DI, [U] assumed layout for (B)."""
import h5py, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = '/home/nishioka/GNN/external_datasets/bam_copv_guidedwave'
REF = B + '/extracted_T25_baseline/25-06-02_08-15-21_GW_Baseline_T25_700bar.h5'
B2  = B + '/extracted_T25_baseline/25-06-02_10-10-03_GW_Baseline_T25_700bar.h5'
ID  = B + '/extracted_T25_id/25-06-13_08-52-27_GW_ID_T25_700bar.h5'
RIDX = [6, 7, 8]; W0 = 250

h = h5py.File(REF, 'r')
chans = h['MetaData/Channels'][()].astype(int)        # (600,2) TX,RX as 1..25 indices
pzt = h['MetaData/Channel_Inputs'][()].ravel().astype(int)  # 25 hardware ids (for labels)
h.close()
IDX = list(range(1, 26))                               # channel indices 1..25
def grid(k): return ((k - 1) // 5, (k - 1) % 5)        # index -> (row,col) row-major

def load_mean(fn):
    h = h5py.File(fn, 'r'); raw = h['Data/Raw_Data']
    s = np.mean([raw[r, :, :].astype(float) for r in RIDX], axis=0)[:, W0:]; h.close(); return s

ref = load_mean(REF)
def di(test):
    d = np.zeros(ref.shape[0])
    for i in range(ref.shape[0]):
        a = ref[i] - ref[i].mean(); b = test[i] - test[i].mean()
        d[i] = 1.0 - float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return d
di_dmg = di(load_mean(ID)); di_ctrl = di(load_mean(B2))

# --- (A) per-PZT involvement: mean DI over all pairs touching each PZT ---
def per_pzt(dvec):
    acc = {k: [] for k in IDX}
    for j, (tx, rx) in enumerate(chans):
        acc[int(tx)].append(dvec[j]); acc[int(rx)].append(dvec[j])
    return {k: np.mean(v) for k, v in acc.items()}
inv_d = per_pzt(di_dmg); inv_c = per_pzt(di_ctrl)
G_d = np.full((5, 5), np.nan); G_c = np.full((5, 5), np.nan)
for k in IDX:
    r, c = grid(k); G_d[r, c] = inv_d[k]; G_c[r, c] = inv_c[k]
# top suspect sensors
order = sorted(IDX, key=lambda k: -inv_d[k])
print('top-5 damage-involved PZT (mean DI):')
for k in order[:5]:
    print('  ch%2d (hw id %2d)  grid%s  DI_dmg=%.3f  DI_ctrl=%.3f'
          % (k, pzt[k - 1], grid(k), inv_d[k], inv_c[k]))

# --- (B) RAPID in assumed uniform grid coords [U] ---
pos = {k: (grid(k)[1] * 1.0, grid(k)[0] * 1.0) for k in IDX}  # (x=col,y=row)
beta = 1.05
gx, gy = np.meshgrid(np.linspace(-0.5, 4.5, 120), np.linspace(-0.5, 4.5, 120))
def rapid(dvec):
    img = np.zeros_like(gx); wsum = np.zeros_like(gx)
    for k, (tx, rx) in enumerate(chans):
        if tx == rx: continue
        (xa, ya), (xb, yb) = pos[int(tx)], pos[int(rx)]
        dab = np.hypot(xb - xa, yb - ya) + 1e-9
        rt = (np.hypot(gx - xa, gy - ya) + np.hypot(gx - xb, gy - yb)) / dab  # elliptical ratio
        w = np.clip((beta - rt) / (beta - 1.0), 0, None)                      # RAPID weight
        img += w * dvec[k]; wsum += w
    return img / (wsum + 1e-9)
R_d = rapid(di_dmg); R_c = rapid(di_ctrl)

# --- figure ---
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
im0 = ax[0].imshow(G_d, origin='lower', cmap='inferno', vmin=0)
for k in IDX:
    r, c = grid(k); ax[0].text(c, r, str(pzt[k - 1]), ha='center', va='center', color='w', fontsize=7)
ax[0].set_title('(A) per-PZT damage involvement [R]\nmean DI of pairs touching each sensor')
ax[0].set_xlabel('grid col'); ax[0].set_ylabel('grid row'); fig.colorbar(im0, ax=ax[0], shrink=0.8)

vmx = max(np.nanmax(R_d), 1e-6)
im1 = ax[1].imshow(R_d, origin='lower', extent=[-0.5, 4.5, -0.5, 4.5], cmap='inferno', vmin=0, vmax=vmx)
ax[1].scatter([pos[p][0] for p in IDX], [pos[p][1] for p in IDX], s=12, c='cyan', marker='s')
ax[1].set_title('(B) RAPID damage image, irreversible [R/U]\n(assumed uniform grid coords [U])')
ax[1].set_xlabel('x (col)'); fig.colorbar(im1, ax=ax[1], shrink=0.8)

im2 = ax[2].imshow(R_c, origin='lower', extent=[-0.5, 4.5, -0.5, 4.5], cmap='inferno', vmin=0, vmax=vmx)
ax[2].scatter([pos[p][0] for p in IDX], [pos[p][1] for p in IDX], s=12, c='cyan', marker='s')
ax[2].set_title('(C) RAPID null map: healthy-control [R]\n(same scale -> should be flat/low)')
ax[2].set_xlabel('x (col)'); fig.colorbar(im2, ax=ax[2], shrink=0.8)
fig.suptitle('BAM COPV guided-wave damage localization (irreversible, 180 kHz)', y=1.03)
fig.tight_layout()
out = '/home/nishioka/Payload2026/copv_gw/copv_damage_localize'
fig.savefig(out + '.png', bbox_inches='tight', dpi=165); fig.savefig(out + '.pdf', bbox_inches='tight')

iy, ix = np.unravel_index(np.argmax(R_d), R_d.shape)
print('\nRAPID peak (damage) at grid ~ (col=%.1f, row=%.1f)' % (gx[iy, ix], gy[iy, ix]))
print('contrast: max(damage RAPID)=%.3f  vs  max(healthy RAPID)=%.3f  ratio=%.1fx'
      % (np.nanmax(R_d), np.nanmax(R_c), np.nanmax(R_d) / (np.nanmax(R_c) + 1e-9)))
print('wrote', out + '.png')
