#!/usr/bin/env python3
"""Rayleigh-Lamb dispersion for the COPV composite wall (CLT-homogenized).

Layup (inner->outer): [CF_90/CF+54/CF-54/CF_90/CF+54/CF-54/CF_90/GFRP_0], H=10mm.
CF  : T700M21    (Dispersion-Calculator MaterialList_TransverselyIsotropic.txt) [S]
GFRP: TVR380M12R (same source) [S]

Method: Classical Laminate Theory -> in-plane A-matrix -> effective axial moduli
(Ex, nxy) -> quasi-isotropic Rayleigh-Lamb dispersion.
Limitation: true anisotropic Lamb requires Christoffel + BC matching per direction.
This isotropic-equivalent is [S] for mesh-sizing and trend; absolute cp calibration
needs the MATLAB Dispersion-Calculator with the full layup.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import math

# ---- Material definitions (transversely isotropic, SI units) [S] ----
# Format: E1, E2 [Pa]; G12 [Pa]; nu12, nu23 [-]; rho [kg/m3]
CF = dict(E1=125.5e9, E2=8.7e9, G12=4.135e9, nu12=0.37, nu23=0.45, rho=1571.0)
GF = dict(E1=46.43e9, E2=14.92e9, G12=5.233e9, nu12=0.27, nu23=0.30, rho=1800.0)


def reduced_Q(mat):
    """Reduced stiffness matrix for UD ply (plane-stress, 1=fiber, 2=transverse)."""
    nu21 = mat['nu12'] * mat['E2'] / mat['E1']
    d = 1.0 - mat['nu12'] * nu21
    return dict(Q11=mat['E1'] / d, Q22=mat['E2'] / d,
                Q12=mat['nu12'] * mat['E2'] / d, Q66=mat['G12'])


def transform_Q(Q, theta_deg):
    """Transformed reduced stiffness Q-bar at angle theta (deg) from local-1 axis."""
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    c2, s2 = c * c, s * s
    c4, s4, c2s2 = c2 * c2, s2 * s2, c2 * s2
    q11, q22, q12, q66 = Q['Q11'], Q['Q22'], Q['Q12'], Q['Q66']
    return dict(
        Q11=(q11 * c4 + 2 * (q12 + 2 * q66) * c2s2 + q22 * s4),
        Q22=(q11 * s4 + 2 * (q12 + 2 * q66) * c2s2 + q22 * c4),
        Q12=((q11 + q22 - 4 * q66) * c2s2 + q12 * (c4 + s4)),
        Q66=((q11 + q22 - 2 * q12 - 2 * q66) * c2s2 + q66 * (c4 + s4)),
    )


# ---- Layup definition: (material_dict, theta_deg, thickness_m) ----
# Same as LAYUP_DEF in generate_copv_gw.py; keep in sync.
LAYUP = [
    (CF,  90.0, 1.0e-3),   # CF hoop inner
    (CF,  54.0, 1.5e-3),   # CF helical +
    (CF, -54.0, 1.5e-3),   # CF helical -
    (CF,  90.0, 1.0e-3),   # CF hoop mid-inner
    (CF,  54.0, 1.5e-3),   # CF helical +
    (CF, -54.0, 1.5e-3),   # CF helical -
    (CF,  90.0, 1.0e-3),   # CF hoop outer
    (GF,   0.0, 1.0e-3),   # GFRP protective outer
]
H_TOTAL = sum(tk for _, _, tk in LAYUP)  # 0.010 m

# ---- CLT A-matrix (in-plane extensional stiffness, Pa*m) ----
A11 = A22 = A12 = A66 = 0.0
rho_sum = 0.0
for mat, theta, tk in LAYUP:
    Q  = reduced_Q(mat)
    Qb = transform_Q(Q, theta)
    A11 += Qb['Q11'] * tk
    A22 += Qb['Q22'] * tk
    A12 += Qb['Q12'] * tk
    A66 += Qb['Q66'] * tk
    rho_sum += mat['rho'] * tk

rho = rho_sum / H_TOTAL   # layer-thickness-weighted average density

det_A = A11 * A22 - A12 ** 2
Ex   = det_A / (A22 * H_TOTAL)   # effective axial modulus (propagation direction)
Ey   = det_A / (A11 * H_TOTAL)   # effective hoop modulus
Gxy  = A66 / H_TOTAL             # effective in-plane shear modulus
nxy  = A12 / A22                  # effective Poisson ratio (axial loaded, hoop contracted)

print('=== CLT A-matrix ===')
print('A11=%.1f  A22=%.1f  A12=%.1f  A66=%.1f  GPa*mm' %
      (A11 / 1e9, A22 / 1e9, A12 / 1e9, A66 / 1e9))
print('=== Effective moduli (axial=x, hoop=y) ===')
print('Ex =%.2f GPa   Ey =%.2f GPa' % (Ex / 1e9, Ey / 1e9))
print('Gxy=%.2f GPa   nxy=%.4f' % (Gxy / 1e9, nxy))
print('rho=%.1f kg/m3' % rho)

# ---- Quasi-isotropic Rayleigh-Lamb dispersion (axial propagation) ----
# Use Ex and nxy as effective isotropic E, nu for the RL solver.
# Implied G = Ex/(2*(1+nxy)) differs from CLT Gxy: accepted approximation for A0 trend.
E  = Ex
nu = nxy
h  = H_TOTAL / 2.0   # half-thickness (m)

G  = E / (2.0 * (1.0 + nu))
cL = math.sqrt(E * (1 - nu) / (rho * (1 + nu) * (1 - 2 * nu)))
cT = math.sqrt(G / rho)
print('\n=== Rayleigh-Lamb solver (isotropic-equivalent with Ex, nxy) ===')
print('cL=%.0f m/s  cT=%.0f m/s  (implied G=%.2f GPa; true CLT Gxy=%.2f GPa)' %
      (cL, cT, G / 1e9, Gxy / 1e9))


def _det_raw(k, w, sym):
    """Pole-free Rayleigh-Lamb determinant; returns real value for root finding."""
    p = np.sqrt((w / cL) ** 2 - k ** 2 + 0j)
    q = np.sqrt((w / cT) ** 2 - k ** 2 + 0j)
    if sym:
        D = (q ** 2 - k ** 2) ** 2 * np.sin(q * h) * np.cos(p * h) \
            + 4 * k ** 2 * p * q * np.cos(q * h) * np.sin(p * h)
    else:
        D = (q ** 2 - k ** 2) ** 2 * np.sin(p * h) * np.cos(q * h) \
            + 4 * k ** 2 * p * q * np.cos(p * h) * np.sin(q * h)
    qq = (w / cT) ** 2 - k ** 2
    return float(np.real(D)) if qq > 0 else float(np.imag(D))


def det_sym(k, w):  return _det_raw(k, w, True)
def det_anti(k, w): return _det_raw(k, w, False)


def all_roots(w, detfn, kmax):
    ks = np.linspace(1.0, kmax, 9000)
    fv = np.array([detfn(k, w) for k in ks])
    roots = []
    for i in range(len(ks) - 1):
        if np.isfinite(fv[i]) and np.isfinite(fv[i + 1]) and fv[i] * fv[i + 1] < 0:
            lo, hi = ks[i], ks[i + 1]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if detfn(lo, w) * detfn(mid, w) <= 0:
                    hi = mid
                else:
                    lo = mid
            roots.append(0.5 * (lo + hi))
    return sorted(roots)


def track(freqs, detfn, start='slow'):
    """Continuation tracking: pick the fundamental mode at each frequency."""
    cps = []
    prev = None
    for f in freqs:
        w    = 2 * np.pi * f
        kmax = w / 250.0
        rts  = all_roots(w, detfn, kmax)
        cand = [w / k for k in rts if k > 0]
        if not cand:
            cps.append(np.nan)
            continue
        if prev is None or not np.isfinite(prev):
            cp = min(cand) if start == 'slow' else max(cand)
        else:
            cp = min(cand, key=lambda c: abs(c - prev))
        cps.append(cp)
        prev = cp
    return np.array(cps)


freqs  = np.linspace(20e3, 350e3, 80)
cp_A0  = track(freqs, det_anti, start='slow')
cp_S0  = track(freqs, det_sym,  start='fast')


def group_vel(freqs, cp):
    w  = 2 * np.pi * freqs
    k  = w / cp
    cg = np.gradient(w, k)
    return cg


cg_A0 = group_vel(freqs, cp_A0)
cg_S0 = group_vel(freqs, cp_S0)

EXC = np.array([60e3, 120e3, 180e3, 260e3, 300e3])
print('\n f[kHz]  | A0: cp[m/s] cg[m/s] lam[mm] | S0: cp cg lam')
print(' --------   --------------------------------   --------------------')
for f in EXC:
    cpA = np.interp(f, freqs, cp_A0); cgA = np.interp(f, freqs, cg_A0)
    cpS = np.interp(f, freqs, cp_S0); cgS = np.interp(f, freqs, cg_S0)
    print('%6.0f  | %6.0f %6.0f %6.2f | %6.0f %6.0f %6.2f' %
          (f / 1e3, cpA, cgA, cpA / f * 1e3, cpS, cgS, cpS / f * 1e3))

print('\n>>> Paste A0 cp values above into generate_copv_gw.py CP[] array.')

lam_min = np.nanmin(np.interp(EXC.max(), freqs, cp_A0)) / EXC.max()
print('\n--- FEM mesh guidance ---')
print('smallest lambda (A0 @ 300kHz) ~ %.2f mm -> element <= %.3f mm (lam/20)' %
      (lam_min * 1e3, lam_min * 1e3 / 20))
print('explicit stable dt ~ elem/cL = %.1f ns (CFL)' % (lam_min / 20 / cL * 1e9))

# ---- Plots ----
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(freqs / 1e3, cp_A0 / 1e3, label='A0')
ax[0].plot(freqs / 1e3, cp_S0 / 1e3, label='S0')
for f in EXC: ax[0].axvline(f / 1e3, c='gray', ls=':', lw=0.7)
ax[0].set_xlabel('frequency [kHz]')
ax[0].set_ylabel('phase velocity [km/s]')
ax[0].set_title('COPV wall Lamb dispersion (CLT Ex-eff.) [S]')
ax[0].legend()
ax[0].set_ylim(0, 8)

lamA = cp_A0 / freqs * 1e3
ax[1].plot(freqs / 1e3, lamA, label=r'A0 $\lambda$')
ax[1].plot(freqs / 1e3, cp_S0 / freqs * 1e3, label=r'S0 $\lambda$')
for f in EXC: ax[1].axvline(f / 1e3, c='gray', ls=':', lw=0.7)
ax[1].set_xlabel('frequency [kHz]')
ax[1].set_ylabel('wavelength [mm]')
ax[1].set_title('wavelength -> FEM mesh sizing')
ax[1].legend()
ax[1].set_ylim(0, 50)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'copv_dispersion.png')
fig.savefig(out, bbox_inches='tight', dpi=180)
fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
print('\nwrote', out)
