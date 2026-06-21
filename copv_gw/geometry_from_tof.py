#!/usr/bin/env python3
"""Recover the 25-PZT array geometry from the guided-wave data itself (no PDF needed).
First-arrival time-of-flight (ToF) between every PZT pair ~ inter-sensor distance;
classical MDS on the 25x25 ToF matrix reconstructs the 2D layout (up to rotation/
reflection/scale). Validates whether the data implies a regular 5x5 grid and gives
the spacing, filling the 'absolute coordinates' gap left by the corrupted doc PDF.
[R] real BAM baseline data; absolute scale via A0 group velocity [U]."""
import h5py, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = '/home/nishioka/GNN/external_datasets/bam_copv_guidedwave'
REF = B + '/extracted_T25_baseline/25-06-02_08-15-21_GW_Baseline_T25_700bar.h5'
RIDX = [6, 7, 8]          # 180 kHz, 3 reps
VG = 2900.0               # A0 group velocity m/s @180 kHz (dispersion_copv.py) [U] -> sets scale only

h = h5py.File(REF, 'r')
fs = h['MetaData/Sampling_Frequency'][()].ravel()[0]; dt = 1.0 / fs
chans = h['MetaData/Channels'][()].astype(int)            # (600,2) TX,RX in 1..25
raw = np.mean([h['Data/Raw_Data'][r, :, :].astype(float) for r in RIDX], axis=0)  # (600,7552)
h.close()

def envelope(sig):
    s = sig - sig.mean()
    F = np.fft.fft(s); n = s.size; hh = np.zeros(n); hh[0] = 1
    hh[1:(n + 1) // 2] = 2
    if n % 2 == 0: hh[n // 2] = 1
    return np.abs(np.fft.ifft(F * hh))

def first_arrival(sig):
    """ToF (s) of the first guided-wave packet: skip the excitation crosstalk, then
    take the time of the FIRST envelope local maximum above 30% of the packet peak
    (group delay of the direct arrival — more stable than a bare threshold)."""
    env = envelope(sig)
    skip = 45                          # ~15 us: past excitation/EM crosstalk
    e = env.copy(); e[:skip] = 0
    pk = e.max()
    if pk <= 0: return np.nan
    thr = 0.30 * pk
    above = np.where(e > thr)[0]
    if above.size == 0: return np.nan
    i0 = above[0]
    # climb to the first local maximum of the first packet
    i = i0
    while i + 1 < e.size and e[i + 1] >= e[i]:
        i += 1
    return i * dt

# 25x25 ToF matrix (symmetric average of i->j and j->i)
N = 25
T = np.full((N, N), np.nan)
for k, (tx, rx) in enumerate(chans):
    t = first_arrival(raw[k])
    i, j = tx - 1, rx - 1
    if np.isnan(T[i, j]): T[i, j] = t
    else: T[i, j] = 0.5 * (T[i, j] + t)
# symmetrize
for i in range(N):
    for j in range(i + 1, N):
        a, b = T[i, j], T[j, i]
        v = np.nanmean([a, b]); T[i, j] = T[j, i] = v
np.fill_diagonal(T, 0.0)
Dist = VG * T * 1e3   # mm  (ToF[s]*vg[m/s]=m -> *1e3 mm)

# --- classical MDS to 3D (PZTs lie on a cylinder surface -> need 3D) ---
D2 = np.nan_to_num(Dist, nan=np.nanmean(Dist)) ** 2
J = np.eye(N) - np.ones((N, N)) / N
Bm = -0.5 * J @ D2 @ J
w, V = np.linalg.eigh(Bm)
order = np.argsort(w)[::-1]; w = w[order]; V = V[:, order]
X3 = V[:, :3] * np.sqrt(np.clip(w[:3], 0, None))      # (25,3) recovered 3D coords (mm)

# --- fit a cylinder: pick the axis (among 3 PCs) whose perpendicular plane gives the
# most circular (constant-radius) cross-section; recovered radius validates vs 176 mm ---
Xc = X3 - X3.mean(0)
U_, S_, Vt_ = np.linalg.svd(Xc, full_matrices=False)   # principal axes = rows of Vt_
best = None
for axis_i in range(3):
    axis = Vt_[axis_i]
    along = Xc @ axis                                   # axial coordinate
    perp = Xc - np.outer(along, axis)                   # in-plane component
    rad = np.linalg.norm(perp, axis=1)
    cv = rad.std() / (rad.mean() + 1e-9)                # circular ring -> low CV
    if best is None or cv < best[0]:
        best = (cv, axis_i, axis, along, perp, rad)
cv, axis_i, axis, along, perp, rad = best
R_recovered = rad.mean()
# unroll: angle around axis in the perpendicular plane
e1 = Vt_[(axis_i + 1) % 3] - (Vt_[(axis_i + 1) % 3] @ axis) * axis; e1 /= np.linalg.norm(e1)
e2 = np.cross(axis, e1)
ang = np.arctan2(perp @ e2, perp @ e1)
unroll = np.column_stack([R_recovered * ang, along])    # (arc-length, axial) mm

def grid(k): return (k // 5, k % 5)
ref = np.array([[grid(k)[1], grid(k)[0]] for k in range(N)], float)
def procrustes(Ain, Bin):
    A = Ain - Ain.mean(0); Bb = Bin - Bin.mean(0)
    Uu, s, Vt = np.linalg.svd(A.T @ Bb); Rr = Uu @ Vt
    return (A @ Rr) * (s.sum() / (A * A).sum()), Rr
Ua, _ = procrustes(unroll, ref)
rc = ref - ref.mean(0)
rmse_grid = np.sqrt(((Ua - rc) ** 2).sum(1)).mean()
nn = [np.sort([np.hypot(*(unroll[i] - unroll[j])) for j in range(N) if j != i])[0] for i in range(N)]
spacing_mm = np.median(nn)

print('ToF matrix: %d/%d pairs' % (np.isfinite(Dist).sum() - N, N * (N - 1)))
print('ToF range %.1f-%.1f us -> distance %.0f-%.0f mm (vg=%.0f [U])'
      % (np.nanmin(T[T > 0]) * 1e6, np.nanmax(T) * 1e6, np.nanmin(Dist[Dist > 0]), np.nanmax(Dist), VG))
print('MDS eigenvalues top5: %s' % np.round(w[:5], 0))
print('  3D explains %.0f%% (2D %.0f%%) -> 3rd dim = cylinder curvature'
      % (100 * w[:3].sum() / w[w > 0].sum(), 100 * w[:2].sum() / w[w > 0].sum()))
print('cylinder fit: radius = %.0f mm   (real vessel OD/2 = 176 mm)   ring-CV=%.2f' % (R_recovered, cv))
print('unrolled layout: spacing ~%.0f mm, grid-RMSE=%.2f (0=perfect 5x5)' % (spacing_mm, rmse_grid))

# --- figure: 3D recovered + unrolled vs ideal grid ---
fig = plt.figure(figsize=(13, 4.4))
axA = fig.add_subplot(1, 3, 1); imA = axA.imshow(Dist, cmap='viridis')
axA.set_title('(a) PZT-PZT distance from ToF [R]\nmm, vg=%.0f m/s [U]' % VG); fig.colorbar(imA, ax=axA, shrink=0.7)
axB = fig.add_subplot(1, 3, 2, projection='3d')
axB.scatter(X3[:, 0], X3[:, 1], X3[:, 2], c=[grid(k)[0] for k in range(N)], cmap='coolwarm', s=40)
axB.set_title('(b) 3D MDS reconstruction [R]\ncylinder R=%.0f mm (real 176)' % R_recovered)
axC = fig.add_subplot(1, 3, 3)
for k in range(N):
    axC.scatter(*Ua[k], s=130, c=[grid(k)[0]], cmap='coolwarm', vmin=0, vmax=4, edgecolor='k', zorder=3)
    axC.text(Ua[k, 0], Ua[k, 1], str(k + 1), ha='center', va='center', fontsize=7, zorder=4)
axC.scatter(rc[:, 0], rc[:, 1], s=190, facecolors='none', edgecolors='gray', lw=0.8, label='ideal 5x5')
axC.set_title('(c) unrolled vs ideal 5x5 [R]\nspacing~%.0f mm, RMSE=%.2f' % (spacing_mm, rmse_grid))
axC.set_aspect('equal'); axC.legend(fontsize=8); axC.grid(alpha=0.3)
fig.suptitle('Data-driven 25-PZT geometry from guided-wave ToF (no PDF needed)', y=1.03)
fig.tight_layout()
out = '/home/nishioka/Payload2026/copv_gw/copv_geometry_from_tof'
fig.savefig(out + '.png', bbox_inches='tight', dpi=160); fig.savefig(out + '.pdf', bbox_inches='tight')
np.savez(out + '.npz', tof=T, dist=Dist, coords3d=X3, unroll=unroll, radius_mm=R_recovered, spacing_mm=spacing_mm)
print('wrote', out + '.png')
