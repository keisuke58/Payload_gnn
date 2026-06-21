# abaqus python extract_damage_di.py   (Abaqus Python: odbAccess + numpy)
"""Extract per-path DI from all damage-library ODB files.

For each COPV_GW_180k_d*.odb in damage_lib/:
  - Load healthy baseline waveforms from COPV_GW_180k.odb (RX node histories)
  - Load damaged waveforms from the damage ODB
  - Compute DI_ij = 1 - corr(damaged_ij, healthy_ij)  for all 24 TX->RX pairs
    (one TX at the fixed centre PZT; therefore 24 paths per simulation)
  - Save as damage_lib/di_library.npz:
        X : (N_damage, 24)   per-path DI
        y : (N_damage, 2)    [circ_mm, axial_mm]
        positions : loaded from damage_positions.json

Note: the current FEM has ONE fixed TX (centre PZT, pzt[12] = (0°, 250mm)).
For a full 600-path library, all 25 TX positions must be simulated separately,
multiplying the job count by 25. This single-TX version gives 24 features
(sufficient for initial validation); a multi-TX extension is noted in the script.
"""
from odbAccess import openOdb
import numpy as np, json, os, glob

HERE    = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(HERE, 'damage_lib')
BASE_ODB = os.path.join(HERE, 'COPV_GW_180k.odb')
POS_JSON = os.path.join(LIB_DIR, 'damage_positions.json')
SKIP     = 250   # skip first N samples (excitation crosstalk, matching BAM processing)


def load_rx_waveforms(odb_path):
    """Return dict {rname: (t, |U|)} for all RX history regions."""
    odb  = openOdb(odb_path, readOnly=True)
    step = odb.steps['gw']
    out  = {}
    for rname, region in step.historyRegions.items():
        outs = region.historyOutputs
        if not any(c in outs for c in ('U1', 'U2', 'U3')):
            continue
        t = None; comp = {}
        for c in ('U1', 'U2', 'U3'):
            if c in outs:
                d = np.array(outs[c].data)
                t = d[:, 0]; comp[c] = d[:, 1]
        u = np.zeros_like(t)
        for c in comp:
            u = u + comp[c] ** 2
        u = np.sqrt(u)[SKIP:]
        out[rname] = u
    odb.close()
    return out


def corr(a, b):
    """Pearson correlation coefficient (scalar)."""
    L = min(len(a), len(b))
    a, b = a[:L], b[:L]
    ma, mb = a.mean(), b.mean()
    num = ((a - ma) * (b - mb)).sum()
    den = (np.sqrt(((a - ma) ** 2).sum()) * np.sqrt(((b - mb) ** 2).sum()))
    return float(num / den) if den > 0 else 1.0


# ─── Load healthy baseline ────────────────────────────────────────────────────
print('Loading healthy baseline: %s' % BASE_ODB)
base_rx = load_rx_waveforms(BASE_ODB)
rnames  = sorted(base_rx.keys())
print('RX regions: %d' % len(rnames))

# ─── Load damage positions ────────────────────────────────────────────────────
positions = json.load(open(POS_JSON))
N = len(positions)
print('Damage positions: %d' % N)

# ─── Process each damage ODB ─────────────────────────────────────────────────
odb_files = sorted(glob.glob(os.path.join(LIB_DIR, 'COPV_GW_180k_d*.odb')))
print('Found %d damage ODB files' % len(odb_files))

X_list, y_list, done_ids = [], [], []
for odb_path in odb_files:
    base = os.path.basename(odb_path)
    idx  = int(base.replace('COPV_GW_180k_d', '').replace('.odb', ''))
    pos  = positions[idx]

    try:
        dam_rx = load_rx_waveforms(odb_path)
    except Exception as e:
        print('  SKIP %s: %s' % (base, e)); continue

    # DI per path (24 paths: centre TX -> each of 24 RX nodes)
    di = np.zeros(len(rnames))
    for k, rn in enumerate(rnames):
        if rn not in dam_rx or rn not in base_rx:
            di[k] = 0.0; continue
        di[k] = 1.0 - corr(dam_rx[rn], base_rx[rn])

    # Normalise by max (consistent with BAM DI computation)
    mx = di.max()
    if mx > 0:
        di = di / mx

    X_list.append(di)
    y_list.append([pos['circ_mm'], pos['axial_mm']])
    done_ids.append(idx)
    print('  %s  circ=%.0f axial=%.0f  di_max=%.3f  di_mean=%.3f'
          % (base, pos['circ_mm'], pos['axial_mm'], di.max(), di.mean()))

if not X_list:
    print('No completed damage ODBs found. Run submit_damage_library.sh first.')
else:
    X = np.array(X_list); y = np.array(y_list)
    out_path = os.path.join(LIB_DIR, 'di_library.npz')
    np.savez(out_path, X=X, y=y, done_ids=np.array(done_ids),
             rnames=np.array(rnames))
    print('\nSaved %s  (X=%s, y=%s)' % (out_path, X.shape, y.shape))
    print('Next: update copv_gw_fem_library.py to load di_library.npz as training data')
