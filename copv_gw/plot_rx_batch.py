#!/usr/bin/env python3
"""Figure for the COPV guided-wave instance: (a) FEM apparent group velocity vs
the A0 dispersion design curve across the 5 real BAM excitation freqs, with the
+14-26% offset shown honestly; (b) a representative RX waveform packet.
Uses system python (rx_*.npz saved by extract_rx_batch.py)."""
import numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = os.path.dirname(__file__) or '.'
FK = [60, 120, 180, 260, 300]
A0_DES = [2145., 2645., 2890., 3058., 3109.]
TXC = None

def far_median_speed(fk):
    z = np.load(os.path.join(D, 'rx_COPV_GW_%dk.npz' % fk))
    txc = z['tx_coord']
    # group RX by region root name
    roots = sorted(set(k.rsplit('_', 1)[0] for k in z.files if k.endswith('_t')))
    vs = []
    for r in roots:
        t = z[r + '_t']
        u = np.sqrt(sum(z[r + '_' + c] ** 2 for c in ('U1', 'U2', 'U3') if (r + '_' + c) in z.files))
        pk = u.max()
        if pk <= 0:
            continue
        ia = int(np.argmax(u > 0.03 * pk))
        ta = t[ia]
        # node coord not stored per-region here; approximate dist via first peak only -> skip dist
        vs.append((ta, pk, u, t))
    return vs

# --- panel (a): dispersion tracking (use summary computed earlier) ---
fem = [2703., 3073., 3556., 3493., 3531.]   # far-field median (extract_rx_batch.py)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(FK, A0_DES, 'o-', color='#2c7fb8', label='A0 analytical (design) [U]')
ax[0].plot(FK, fem, 's--', color='#d7301f', label='FEM apparent (first-arrival)')
for x, y0, y1 in zip(FK, A0_DES, fem):
    ax[0].annotate('+%.0f%%' % (100 * (y1 - y0) / y0), (x, y1), textcoords='offset points',
                   xytext=(0, 6), ha='center', fontsize=7, color='#d7301f')
ax[0].set_xlabel('excitation frequency [kHz]'); ax[0].set_ylabel('group velocity [m/s]')
ax[0].set_title('(a) A0 dispersion: FEM tracks trend,\n+14-26% high (S0-contaminated pick) [S]')
ax[0].legend(fontsize=8); ax[0].set_ylim(1800, 4000); ax[0].grid(alpha=0.3)

# --- panel (b): representative RX packet at 180 kHz ---
z = np.load(os.path.join(D, 'rx_COPV_GW_180k.npz'))
roots = sorted(set(k.rsplit('_', 1)[0] for k in z.files if k.endswith('_t')))
# pick a mid-distance RX with a clean packet (largest peak among first few)
best = None
for r in roots:
    t = z[r + '_t']
    u = np.sqrt(sum(z[r + '_' + c] ** 2 for c in ('U1', 'U2', 'U3') if (r + '_' + c) in z.files))
    if best is None or u.max() > best[2]:
        best = (r, t, u.max(), u)
r, t, pkmax, u = best
ax[1].plot(t * 1e3, u * 1e6, color='#2c7fb8', lw=0.9)
ax[1].set_xlabel('time [ms]'); ax[1].set_ylabel(r'|U| [$\mu$m]')
ax[1].set_title('(b) representative RX packet @180 kHz\n(%s)' % r.replace('Node ', ''))
ax[1].grid(alpha=0.3)

fig.suptitle('COPV Type-IV guided-wave FEM (5 real BAM excitation freqs, 90deg sector) [U material]', y=1.04, fontsize=10)
out = os.path.join(D, 'copv_gw_validation')
fig.savefig(out + '.png', bbox_inches='tight', dpi=180)
fig.savefig(out + '.pdf', bbox_inches='tight')
print('wrote', out + '.png/.pdf')
