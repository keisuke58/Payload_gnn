#!/usr/bin/env python3
"""
gen_ogw_phase0_inp.py
OGW Phase0 FEM: CFRP flat plate (no stringer), Abaqus/Explicit guided wave
----------------------------------------------------------------------
Goal: match A0/S0 group velocity to OGW healthy measurement for material calibration
Domain: 300×300mm, S4R 1mm, [45/0/-45/90]s 8-ply 2mm CFRP
Excitation: 5-cycle Hann toneburst @ 100kHz, concentrated force at center
Output: sensor history @ 7.8125e-7 s (= OGW dt), field @ 20e-6 s

Usage:
  python scripts/gen_ogw_phase0_inp.py [--freq 50|100|200|300] [--size 300]
  → abaqus_work/ogw_fem/OGW_Phase0_Healthy_<FREQ>kHz.inp
"""
import argparse, os, math, numpy as np

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--freq',  type=int, default=100,   help='Toneburst frequency in kHz')
ap.add_argument('--size',  type=int, default=300,   help='Domain side length in mm')
ap.add_argument('--dx',    type=float, default=1.0, help='Element size in mm')
ap.add_argument('--ncyc', type=int, default=5,      help='Number of Hann cycles')
args = ap.parse_args()

FREQ_kHz = args.freq
FREQ     = FREQ_kHz * 1e3          # Hz
N_CYC    = args.ncyc
LSIZE    = float(args.size)        # mm, domain side
DX       = args.dx                 # mm, element size
T_TOTAL  = 8.0e-4                  # s, 0.8ms (match OGW)
DT_HIST  = 7.8125e-7               # s, 1.28 MHz (OGW sampling rate)
DT_FIELD = 20.0e-6                 # s, field output every 20µs (40 frames)

PLY_T  = 0.25                      # mm per ply
LAYUP  = [45., 0., -45., 90., 90., -45., 0., 45.]   # [45/0/-45/90]s
MAT    = 'CFRP_T1000G'

# Element resolution check: need DX <= lambda_A0 / 10
# A0 @ FREQ in 2mm CFRP ~ 1500 m/s (conservative), lambda = c/f
c_A0_est = 1500.0  # m/s
lam_est  = c_A0_est / FREQ * 1000  # mm
if DX > lam_est / 8:
    print(f'WARNING: DX={DX}mm may be coarse for {FREQ_kHz}kHz '
          f'(λ_A0 est={lam_est:.1f}mm, λ/8={lam_est/8:.2f}mm)')

NX = int(LSIZE / DX)   # grid divisions along X
NY = int(LSIZE / DX)   # grid divisions along Y
N_NODES = (NX + 1) * (NY + 1)
N_ELEMS = NX * NY

def nid(i, j):
    """Node ID: row i (0..NX), col j (0..NY). 1-based."""
    return i * (NY + 1) + j + 1

# Excitation: center of plate
EXC_I, EXC_J = NX // 2, NY // 2
EXC_NID = nid(EXC_I, EXC_J)

# Sensors along +X from excitation (every 30mm, up to 150mm or domain edge)
# and along +Y (same)
SENSOR_SPACING_MM = 30
sensors = {}  # name → node_id
for dist_mm in range(SENSOR_SPACING_MM, int(LSIZE/2) + 1, SENSOR_SPACING_MM):
    di = int(dist_mm / DX)
    # +X
    i_x = EXC_I + di
    if i_x <= NX:
        sensors[f'Sx_p{dist_mm}mm'] = nid(i_x, EXC_J)
    # -X
    i_xm = EXC_I - di
    if i_xm >= 0:
        sensors[f'Sx_m{dist_mm}mm'] = nid(i_xm, EXC_J)
    # +Y
    j_y = EXC_J + di
    if j_y <= NY:
        sensors[f'Sy_p{dist_mm}mm'] = nid(EXC_I, j_y)
    # +45° diagonal (for anisotropy check)
    if i_x <= NX and j_y <= NY:
        sensors[f'Sd_{dist_mm}mm'] = nid(i_x, j_y)

# ── Hann toneburst amplitude table ────────────────────────────────────────────
def hann_toneburst_table(freq, n_cycles, dt=5e-7):
    """Returns (times, amplitudes) for a Hann-windowed toneburst."""
    T_burst = n_cycles / freq
    t = np.arange(0, T_burst + dt * 0.5, dt)
    amp = 0.5 * (1.0 - np.cos(2*np.pi*freq*t/n_cycles)) * np.sin(2*np.pi*freq*t)
    # trailing zero
    t   = np.append(t, t[-1] + dt)
    amp = np.append(amp, 0.0)
    return t, amp

t_amp, amp = hann_toneburst_table(FREQ, N_CYC)

# ── output path ───────────────────────────────────────────────────────────────
JOB = f'OGW_Phase0_Healthy_{FREQ_kHz}kHz'
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'abaqus_work', 'ogw_fem')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f'{JOB}.inp')

print(f'OGW Phase0 inp generator')
print(f'  Domain:  {LSIZE}×{LSIZE}mm, DX={DX}mm')
print(f'  Mesh:    {N_NODES:,} nodes, {N_ELEMS:,} S4R elements')
print(f'  Layup:   {LAYUP} ({len(LAYUP)}×{PLY_T}mm = {len(LAYUP)*PLY_T}mm total)')
print(f'  Freq:    {FREQ_kHz}kHz, {N_CYC}-cycle Hann')
print(f'  Time:    {T_TOTAL*1e3:.1f}ms, hist Δt={DT_HIST*1e6:.4f}µs, field Δt={DT_FIELD*1e6:.0f}µs')
print(f'  Sensors: {len(sensors)}')
print(f'  Output:  {out_path}')

# ── write inp ─────────────────────────────────────────────────────────────────
with open(out_path, 'w') as f:
    W = f.write

    def line(*args):
        W(' '.join(str(a) for a in args) + '\n')

    def comment(txt=''):
        W(f'** {txt}\n')

    # ── header ────────────────────────────────────────────────────────────────
    comment(f'Job: {JOB}')
    comment(f'OGW omega-stringer Phase0 — flat CFRP plate, material calibration')
    comment(f'Domain {LSIZE}x{LSIZE}mm | DX={DX}mm | f={FREQ_kHz}kHz | t={T_TOTAL*1e3:.1f}ms')
    comment(f'Nodes={N_NODES} | Elems={N_ELEMS} | Layup=[45/0/-45/90]s 2mm')
    W('*Preprint, echo=NO, model=NO, history=NO, contact=NO\n')

    # ── empty part ────────────────────────────────────────────────────────────
    comment()
    comment('PART (orphan mesh defined in instance)')
    comment()
    W('*Part, name=OGW_Plate\n')
    W('*End Part\n')

    # ── assembly → instance ───────────────────────────────────────────────────
    comment()
    comment('ASSEMBLY')
    comment()
    W('*Assembly, name=Assembly\n')
    comment()
    W('*Instance, name=OGW_Plate-1, part=OGW_Plate\n')

    # Nodes (defined inside instance = orphan mesh)
    W('*Node\n')
    for i in range(NX + 1):
        x = -LSIZE/2.0 + i * DX
        for j in range(NY + 1):
            y = -LSIZE/2.0 + j * DX
            W(f'{nid(i,j):8d}, {x:12.4f}, {y:12.4f},        0.\n')

    # Elements (S4R)
    W('*Element, type=S4R\n')
    eid = 1
    for i in range(NX):
        for j in range(NY):
            n1 = nid(i,   j  )
            n2 = nid(i+1, j  )
            n3 = nid(i+1, j+1)
            n4 = nid(i,   j+1)
            W(f'{eid:8d}, {n1:8d}, {n2:8d}, {n3:8d}, {n4:8d}\n')
            eid += 1

    # Sets inside instance
    W(f'*Nset, nset=Set-All, generate\n')
    W(f'     1, {N_NODES},      1\n')
    W(f'*Elset, elset=Set-All, generate\n')
    W(f'     1, {N_ELEMS},      1\n')

    # Composite shell section [45/0/-45/90]s
    comment(f'Section: CFRP quasi-isotropic {len(LAYUP)}x{PLY_T}mm = {len(LAYUP)*PLY_T}mm')
    W('*Shell Section, elset=Set-All, composite\n')
    for angle in LAYUP:
        W(f'{PLY_T}, 3, {MAT}, {angle}\n')

    W('*End Instance\n')
    comment()

    # Assembly-level Nsets (reference instance)
    W(f'*Nset, nset=Set-Excitation, instance=OGW_Plate-1\n')
    W(f' {EXC_NID},\n')
    for sname, snid_val in sensors.items():
        W(f'*Nset, nset={sname}, instance=OGW_Plate-1\n')
        W(f' {snid_val},\n')

    W('*End Assembly\n')

    # ── amplitude ─────────────────────────────────────────────────────────────
    comment()
    comment(f'AMPLITUDE: {N_CYC}-cycle Hann @ {FREQ_kHz}kHz')
    comment()
    W('*Amplitude, name=Amp-ToneBurst\n')
    pairs = list(zip(t_amp, amp))
    for k in range(0, len(pairs), 4):
        chunk = pairs[k:k+4]
        W(','.join(f'{t:.6e}, {a:.10f}' for t, a in chunk) + '\n')

    # ── materials ─────────────────────────────────────────────────────────────
    comment()
    comment('MATERIALS')
    comment('Initial T1000G values — calibrate E1,E2,G12 to OGW A0/S0 dispersion')
    comment()
    W(f'*Material, name={MAT}\n')
    W('*Density\n')
    W(' 1.6e-09,\n')                   # t/mm³
    W('*Elastic, type=LAMINA\n')
    W('160000., 10000.,  0.3, 5000., 5000., 3000.\n')  # MPa: E1,E2,nu12,G12,G13,G23

    # ── step ──────────────────────────────────────────────────────────────────
    comment()
    comment('STEP: guided wave propagation')
    comment()
    W('*Step, name=Step-Wave, nlgeom=NO\n')
    W('*Dynamic, Explicit\n')
    W(f', {T_TOTAL}\n')
    W('*Bulk Viscosity\n')
    W('0.06, 1.2\n')
    comment()
    comment('LOADS')
    comment()
    W('** Concentrated force at plate center → Z direction (face-normal)\n')
    W('*Cload, amplitude=Amp-ToneBurst\n')
    W('Set-Excitation, 3, 1.\n')
    comment()
    comment('OUTPUT')
    comment()
    W('*Restart, write, number interval=1, time marks=NO\n')
    comment('Field: full-field U3 every 20µs for snapshot comparison with OGW SLDV')
    W(f'*Output, field, time interval={DT_FIELD}\n')
    W('*Node Output\n')
    W('U3,\n')
    comment('History: sensors at OGW sampling rate (7.8125e-7 s) for group velocity')
    W(f'*Output, history, time interval={DT_HIST}\n')
    for sname in sensors:
        W(f'*Node Output, nset={sname}\n')
        W('U3,\n')
    W('*End Step\n')

# ── PBS submit script ──────────────────────────────────────────────────────────
pbs_path = os.path.join(out_dir, f'submit_{JOB}.sh')
with open(pbs_path, 'w') as f:
    f.write(f"""#!/bin/bash
#PBS -N {JOB}
#PBS -l nodes=1:ppn=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o {out_dir}/{JOB}.log

cd {out_dir}
module load abaqus 2>/dev/null || true

abaqus job={JOB} input={JOB}.inp cpus=1 interactive > {JOB}_stdout.txt 2>&1
echo "Exit code: $?" >> {JOB}_stdout.txt
""")
os.chmod(pbs_path, 0o755)

print()
print(f'Generated:')
print(f'  inp: {out_path}  ({os.path.getsize(out_path)/1024/1024:.1f} MB)')
print(f'  pbs: {pbs_path}')
print()
print('Next steps:')
print(f'  1. DATACHECK: abaqus job={JOB} input={JOB}.inp datacheck cpus=1 interactive')
print(f'  2. Submit:    qsub {pbs_path}')
print(f'  3. After run: extract U3 → compare A0 group velocity with OGW healthy signal')
