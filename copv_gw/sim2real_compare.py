#!/usr/bin/env python3
"""First sim2real comparison: real BAM COPV guided-wave (180 kHz, healthy/700bar/T25)
vs my Abaqus/Explicit COPV sim (180 kHz). Geometry-independent checks (dominant
frequency, packet character) because the real PZT spacing is still in the scanned
PDF -> absolute time-of-flight is NOT calibrated here (stated honestly).
[R] real BAM data, [S] sim self-consistent, [U] representative material/spacing."""
import h5py, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BAM = '/home/nishioka/GNN/external_datasets/bam_copv_guidedwave/extracted_T25_baseline/25-06-02_08-15-21_GW_Baseline_T25_700bar.h5'
SIM = '/home/nishioka/Payload2026/copv_gw/rx_COPV_GW_180k.npz'
FREQ = 180e3
RAW_IDX_180 = 6          # 0-based: indices 7,8,9 (1-based) are 180 kHz; take first

# ---- real ----
h = h5py.File(BAM, 'r')
dt_r = 1.0 / h['MetaData/Sampling_Frequency'][()].ravel()[0]
chans = h['MetaData/Channels'][()].astype(int)     # (600,2) TX,RX
def pair_index(tx, rx):
    m = np.where((chans[:, 0] == tx) & (chans[:, 1] == rx))[0]
    return int(m[0]) if m.size else None
# adjacent (TX1->RX2) and far-in-row (TX1->RX5)
pi_near = pair_index(1, 2); pi_far = pair_index(1, 5)
raw = h['Data/Raw_Data']
sig_near = raw[RAW_IDX_180, pi_near, :].astype(float)
sig_far = raw[RAW_IDX_180, pi_far, :].astype(float)
t_r = np.arange(sig_near.size) * dt_r
exc = h['MetaData/Signal_Data_Burst'][()][2]        # row2 = 180 kHz excitation
h.close()

# ---- sim ----
# Use a SIGNED displacement component (oscillates at the 180 kHz carrier), NOT the
# magnitude |U| (its rectified envelope peaks at ~kHz and hides the carrier).
z = np.load(SIM)
roots = sorted(set(k.rsplit('_', 1)[0] for k in z.files if k.endswith('_t')))
best = None
for r in roots:
    t = z[r + '_t']
    # pick the signed component with the largest peak-to-peak for this node
    comp_best = None
    for c in ('U1', 'U2', 'U3'):
        key = r + '_' + c
        if key in z.files:
            sig = z[key]
            ptp = sig.max() - sig.min()
            if comp_best is None or ptp > comp_best[1]:
                comp_best = (sig, ptp)
    if comp_best is None:
        continue
    if best is None or comp_best[1] > best[2]:
        best = (r, t, comp_best[1], comp_best[0])
rname, t_s, _, u_s = best        # u_s is now a SIGNED carrier-bearing component

def spec(sig, dt):
    sig = sig - sig.mean()
    w = np.hanning(sig.size)
    S = np.abs(np.fft.rfft(sig * w))
    fr = np.fft.rfftfreq(sig.size, dt)
    return fr, S / (S.max() + 1e-30)

fr_r, S_r = spec(sig_near, dt_r)
fr_s, S_s = spec(u_s, t_s[1] - t_s[0])
# dominant freq
fpk_r = fr_r[np.argmax(S_r)]; fpk_s = fr_s[np.argmax(S_s)]

# ---- plot ----
fig, ax = plt.subplots(2, 2, figsize=(11, 6.4))
ax[0, 0].plot(t_r * 1e3, sig_near / np.abs(sig_near).max(), lw=0.6, color='#222')
ax[0, 0].set_title('(a) REAL BAM RX  TX1->RX2  180 kHz  700bar T25 [R]')
ax[0, 0].set_xlabel('time [ms]'); ax[0, 0].set_ylabel('norm. amp'); ax[0, 0].grid(alpha=0.3)
ax[0, 0].set_xlim(0, t_r.max() * 1e3)

snorm = np.abs(u_s).max() + 1e-30
ax[0, 1].plot(t_s * 1e3, u_s / snorm, lw=0.8, color='#2c7fb8')
ax[0, 1].set_title('(b) SIM Abaqus RX  180 kHz  90deg sector [S/U]\n(%s)' % rname.replace('Node ', ''))
ax[0, 1].set_xlabel('time [ms]'); ax[0, 1].set_ylabel('norm. signed U'); ax[0, 1].grid(alpha=0.3)

# analytic-signal envelope (works for signed carrier signals)
def env(sig):
    from numpy.fft import fft, ifft, fftfreq
    F = fft(sig - sig.mean())
    h = np.zeros(sig.size); n = sig.size
    h[0] = 1; h[1:(n + 1) // 2] = 2
    if n % 2 == 0:
        h[n // 2] = 1
    return np.abs(ifft(F * h))
ax[1, 0].plot(t_r * 1e3, env(sig_near) / env(sig_near).max(), lw=0.8, color='#222', label='real env')
ax[1, 0].plot(t_s * 1e3, env(u_s) / (env(u_s).max() + 1e-30), lw=0.9, color='#2c7fb8', label='sim env')
ax[1, 0].set_xlim(0, 0.6); ax[1, 0].set_title('(c) packet envelope, first 0.6 ms\n(time axes NOT geometry-aligned [U])')
ax[1, 0].set_xlabel('time [ms]'); ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

ax[1, 1].plot(fr_r / 1e3, S_r, lw=0.9, color='#222', label='real (peak %.0f kHz)' % (fpk_r / 1e3))
ax[1, 1].plot(fr_s / 1e3, S_s, lw=1.0, color='#2c7fb8', label='sim (peak %.0f kHz)' % (fpk_s / 1e3))
ax[1, 1].axvline(FREQ / 1e3, color='#d7301f', ls=':', lw=1, label='excitation 180 kHz')
ax[1, 1].set_xlim(0, 600); ax[1, 1].set_title('(d) spectra: dominant freq match [R vs S]')
ax[1, 1].set_xlabel('frequency [kHz]'); ax[1, 1].set_ylabel('norm. |FFT|'); ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)

fig.suptitle('sim2real COPV guided wave @180 kHz: real BAM PZT vs Abaqus/Explicit', y=1.01, fontsize=11)
fig.tight_layout()
out = '/home/nishioka/Payload2026/copv_gw/sim2real_180k'
fig.savefig(out + '.png', bbox_inches='tight', dpi=170)
fig.savefig(out + '.pdf', bbox_inches='tight')

print('REAL: dt=%.3e s, %d samples, record %.2f ms' % (dt_r, sig_near.size, t_r.max() * 1e3))
print('  pairs: TX1->RX2 idx=%s, TX1->RX5 idx=%s' % (pi_near, pi_far))
print('  real peak-freq = %.0f kHz (excit 180)' % (fpk_r / 1e3))
print('SIM : dt=%.3e s, %d samples, record %.3f ms' % (t_s[1] - t_s[0], u_s.size, t_s.max() * 1e3))
print('  sim peak-freq  = %.0f kHz' % (fpk_s / 1e3))
print('  sim RX used    = %s' % rname)
print('wrote', out + '.png')
