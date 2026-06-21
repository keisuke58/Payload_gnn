#!/usr/bin/env python3
"""COPV damage localization on the TRUE sensor geometry (from the EWSHM2024 dataset
paper, El Moutaouakil et al., doi:10.58286/29754, CC BY 4.0): 5 rings x 5 PZT,
circumferential spacing 221 mm (=72deg, full wrap of the 1105 mm circumference),
axial ring pitch 312.5 mm. RAPID on the unrolled cylinder with circumferential
wrap-around. [R] real BAM data + [R] published geometry (no more [U] for layout)."""
import h5py, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = '/home/nishioka/GNN/external_datasets/bam_copv_guidedwave'
REF = B + '/extracted_T25_baseline/25-06-02_08-15-21_GW_Baseline_T25_700bar.h5'
ID  = B + '/extracted_T25_id/25-06-13_08-52-27_GW_ID_T25_700bar.h5'
RIDX = [6, 7, 8]; W0 = 250
# --- TRUE geometry (paper) ---
CIRC = 221.0          # mm circumferential PZT spacing
RING = 312.5          # mm axial ring pitch
P = 5 * CIRC          # 1105 mm circumference (period for wrap-around)
def pos(k):           # channel index 1..25 -> (arc_mm, axial_mm); ring=(k-1)//5, circ=(k-1)%5
    r, c = (k - 1) // 5, (k - 1) % 5
    return (c * CIRC, r * RING)

h = h5py.File(REF, 'r')
chans = h['MetaData/Channels'][()].astype(int)
ref = np.mean([h['Data/Raw_Data'][r, :, :].astype(float) for r in RIDX], axis=0)[:, W0:]
h.close()
def load(fn):
    g = h5py.File(fn, 'r'); s = np.mean([g['Data/Raw_Data'][r, :, :].astype(float) for r in RIDX], axis=0)[:, W0:]; g.close(); return s
def di(test):
    d = np.zeros(ref.shape[0])
    for i in range(ref.shape[0]):
        a = ref[i] - ref[i].mean(); b = test[i] - test[i].mean()
        d[i] = 1.0 - float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return d
di_id = di(load(ID))

IDX = list(range(1, 26))
# (A) per-PZT involvement on true grid
acc = {k: [] for k in IDX}
for j, (tx, rx) in enumerate(chans):
    acc[int(tx)].append(di_id[j]); acc[int(rx)].append(di_id[j])
inv = {k: np.mean(v) for k, v in acc.items()}
order = sorted(IDX, key=lambda k: -inv[k])
print('TRUE geometry: circ=%.0fmm (72deg), ring=%.1fmm, circumference=%.0fmm' % (CIRC, RING, P))
print('top-5 damage-involved PZT (mean DI):')
for k in order[:5]:
    a, z = pos(k); print('  sensor %2d  ring%d/circ%d  arc=%.0fmm axial=%.0fmm  DI=%.3f' % (k, (k-1)//5, (k-1)%5, a, z, inv[k]))

# (B) RAPID on unrolled cylinder, circumferential wrap-around
beta = 1.10
gx, gy = np.meshgrid(np.linspace(0, P, 200), np.linspace(-150, 4 * RING + 150, 160))
def wrapdx(x1, x2):
    d = np.abs(x1 - x2); return np.minimum(d, P - d)
def rapid(dvec):
    img = np.zeros_like(gx); wsum = np.zeros_like(gx)
    for j, (tx, rx) in enumerate(chans):
        (xa, ya), (xb, yb) = pos(int(tx)), pos(int(rx))
        dab = np.hypot(wrapdx(xb, xa), yb - ya) + 1e-9
        rA = np.hypot(wrapdx(gx, xa), gy - ya); rB = np.hypot(wrapdx(gx, xb), gy - yb)
        rt = (rA + rB) / dab
        w = np.clip((beta - rt) / (beta - 1.0), 0, None)
        img += w * dvec[j]; wsum += w
    return img / (wsum + 1e-9)
R = rapid(di_id)
iy, ix = np.unravel_index(np.argmax(R), R.shape)
print('\nRAPID damage peak at arc=%.0f mm (%.0f deg), axial=%.0f mm' % (gx[iy, ix], gx[iy, ix] / P * 360, gy[iy, ix]))
print('  (D1 paper-clue: near sensors 5 & 6 path; sensor5=ring0/circ4 arc=%.0f, sensor6=ring1/circ0 arc=0)' % (4 * CIRC))

# --- figure: unrolled vessel ---
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
im = ax[1].imshow(R, origin='lower', extent=[0, P, -150, 4 * RING + 150], aspect='auto', cmap='inferno')
for k in IDX:
    a, z = pos(k); ax[1].scatter(a, z, s=90, facecolors='none', edgecolors='cyan', lw=1.2)
    ax[1].text(a, z, str(k), color='cyan', fontsize=6, ha='center', va='center')
ax[1].set_xlabel('circumferential arc [mm] (0-1105 = full wrap)'); ax[1].set_ylabel('axial [mm]')
ax[1].set_title('(B) RAPID damage image on TRUE unrolled cylinder [R]\npeak arc=%.0fmm axial=%.0fmm' % (gx[iy, ix], gy[iy, ix]))
fig.colorbar(im, ax=ax[1], shrink=0.8)
# per-PZT grid
G = np.full((5, 5), np.nan)
for k in IDX: G[(k - 1) // 5, (k - 1) % 5] = inv[k]
im0 = ax[0].imshow(G, origin='lower', cmap='inferno', extent=[-0.5, 4.5, -0.5, 4.5])
for k in IDX:
    ax[0].text((k - 1) % 5, (k - 1) // 5, str(k), color='cyan', ha='center', va='center', fontsize=8)
ax[0].set_xlabel('circumferential position (x221 mm)'); ax[0].set_ylabel('ring (axial, x312.5 mm)')
ax[0].set_title('(A) per-PZT damage involvement [R]\n(true 5x5 ring x circumference)')
fig.colorbar(im0, ax=ax[0], shrink=0.8)
fig.suptitle('COPV irreversible-damage localization on published BAM sensor geometry (180 kHz)', y=1.02)
fig.tight_layout()
out = '/home/nishioka/Payload2026/copv_gw/copv_localize_true'
fig.savefig(out + '.png', bbox_inches='tight', dpi=160); fig.savefig(out + '.pdf', bbox_inches='tight')
print('wrote', out + '.png')
