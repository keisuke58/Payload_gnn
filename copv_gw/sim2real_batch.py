#!/usr/bin/env python3
"""Band-wide sim2real: real BAM COPV PZT vs Abaqus sim across all 5 excitation
freqs (60/120/180/260/300 kHz, healthy/700bar/T25). Per freq: dominant frequency,
spectral centroid, and a normalized spectral cosine-overlap. Geometry-independent
(ToF not calibrated; real PZT spacing pending PDF). [R] real, [S] sim, [U] material."""
import h5py, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BAM = '/home/nishioka/GNN/external_datasets/bam_copv_guidedwave/extracted_T25_baseline/25-06-02_08-15-21_GW_Baseline_T25_700bar.h5'
D = '/home/nishioka/Payload2026/copv_gw'
# freq -> (raw-data 0-based index of first rep, sim npz job tag)
FREQS = [(60, 0, '60k'), (120, 3, '120k'), (180, 6, '180k'), (260, 9, '260k'), (300, 12, '300k')]

h = h5py.File(BAM, 'r')
dt_r = 1.0 / h['MetaData/Sampling_Frequency'][()].ravel()[0]
chans = h['MetaData/Channels'][()].astype(int)
pi = int(np.where((chans[:, 0] == 1) & (chans[:, 1] == 2))[0][0])   # TX1->RX2

def spec(sig, dt, fmin=20e3, fmax=600e3):
    # exclude <20 kHz: the real record has a large DC/baseline + excitation-crosstalk
    # spike there that is instrumentation, not guided-wave content (GW lives 60-300 kHz).
    sig = sig - sig.mean()
    S = np.abs(np.fft.rfft(sig * np.hanning(sig.size)))
    fr = np.fft.rfftfreq(sig.size, dt)
    m = (fr >= fmin) & (fr <= fmax)
    return fr[m], S[m] / (S[m].max() + 1e-30)

def sim_signed(job):
    z = np.load(os.path.join(D, 'rx_COPV_GW_%s.npz' % job))
    roots = sorted(set(k.rsplit('_', 1)[0] for k in z.files if k.endswith('_t')))
    best = None
    for r in roots:
        for c in ('U1', 'U2', 'U3'):
            key = r + '_' + c
            if key in z.files:
                sig = z[key]; ptp = sig.max() - sig.min()
                if best is None or ptp > best[2]:
                    best = (r, z[r + '_t'], ptp, sig)
    return best[1], best[3]   # t, signed component

def centroid(fr, S):
    return float((fr * S).sum() / (S.sum() + 1e-30))

# common grid for cosine overlap
grid = np.linspace(20e3, 600e3, 1200)
rows = []
fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), sharey=True)
for (fk, ridx, job), ax in zip(FREQS, axes):
    real = h['Data/Raw_Data'][ridx, pi, :].astype(float)
    fr_r, S_r = spec(real, dt_r)
    t_s, u_s = sim_signed(job)
    fr_s, S_s = spec(u_s, t_s[1] - t_s[0])
    dr = fr_r[np.argmax(S_r)] / 1e3; ds = fr_s[np.argmax(S_s)] / 1e3
    cr = centroid(fr_r, S_r) / 1e3; cs = centroid(fr_s, S_s) / 1e3
    gr = np.interp(grid, fr_r, S_r); gs = np.interp(grid, fr_s, S_s)
    cos = float((gr * gs).sum() / (np.linalg.norm(gr) * np.linalg.norm(gs) + 1e-30))
    rows.append((fk, dr, ds, cr, cs, cos))
    ax.plot(fr_r / 1e3, S_r, color='#222', lw=0.9, label='real [R]')
    ax.plot(fr_s / 1e3, S_s, color='#2c7fb8', lw=1.0, label='sim [S]')
    ax.axvline(fk, color='#d7301f', ls=':', lw=1)
    ax.set_title('%d kHz\nreal %.0f / sim %.0f kHz\ncos=%.2f' % (fk, dr, ds, cos), fontsize=9)
    ax.set_xlabel('freq [kHz]'); ax.set_xlim(0, 600); ax.grid(alpha=0.3)
axes[0].set_ylabel('norm |FFT|'); axes[0].legend(fontsize=8)
fig.suptitle('Band-wide sim2real: real BAM PZT vs Abaqus COPV sim (TX1->RX2, healthy) [U material]', y=1.06)
fig.tight_layout()
out = os.path.join(D, 'sim2real_allfreq')
fig.savefig(out + '.png', bbox_inches='tight', dpi=160)
fig.savefig(out + '.pdf', bbox_inches='tight')
h.close()

print('%5s | real_pk sim_pk | real_cen sim_cen | cos-overlap' % 'kHz')
for fk, dr, ds, cr, cs, cos in rows:
    print('%5d |  %5.0f  %5.0f |   %5.0f   %5.0f  |  %.3f' % (fk, dr, ds, cr, cs, cos))
print('\nexcitation-band tracking: real & sim dominant freq should both rise with excitation;')
print('cos-overlap is a geometry-free spectral-agreement score (1=identical shape).')
print('wrote', out + '.png')
