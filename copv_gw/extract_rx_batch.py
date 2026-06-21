# abq2024 python extract_rx_batch.py
# Extract RX histories from all 5 frequency odbs, save per-freq npz, and test
# whether the FEM apparent group velocity tracks the A0 dispersion design curve.
from odbAccess import openOdb
import numpy as np

JOBS = [(60, 'COPV_GW_60k'), (120, 'COPV_GW_120k'), (180, 'COPV_GW_180k'),
        (260, 'COPV_GW_260k'), (300, 'COPV_GW_300k')]
# A0 group-velocity design table (dispersion_copv.py, CLT orthotropic layup) [S] 2026-06-21
# Old homogenized E=50GPa values were: {60:2145, 120:2645, 180:2890, 260:3058, 300:3109}
A0_DESIGN = {60: 2054., 120: 2081., 180: 2043., 260: 1996., 300: 1977.}

def extract(job):
    odb = openOdb(job + '.odb', readOnly=True)
    step = odb.steps['gw']
    txc = np.array(odb.rootAssembly.nodeSets['TX'].nodes[0][0].coordinates)
    rows = []; store = {}
    for rname, region in step.historyRegions.items():
        outs = region.historyOutputs
        if not any(k in outs for k in ('U1', 'U2', 'U3')):
            continue
        t = None; comp = {}
        for c in ('U1', 'U2', 'U3'):
            if c in outs:
                d = np.array(outs[c].data); t = d[:, 0]; comp[c] = d[:, 1]
        u = np.sqrt(sum(comp[c] ** 2 for c in comp))
        try:
            cc = np.array(region.point.node.coordinates)
            dist = np.sqrt(((cc - txc) ** 2).sum())
        except Exception:
            dist = np.nan
        pk = u.max()
        thr = 0.03 * pk if pk > 0 else 0
        ia = int(np.argmax(u > thr)) if pk > 0 else -1
        ta = t[ia] if ia >= 0 else np.nan
        v = (dist * 1e-3) / ta if (ta and ta > 0 and np.isfinite(dist)) else np.nan
        rows.append((dist, pk, ta, v))
        store[rname + '_t'] = t
        for c in comp:
            store[rname + '_' + c] = comp[c]
    np.savez('rx_%s.npz' % job, tx_coord=txc, **store)
    odb.close()
    return rows

print('%5s %6s %9s %9s %9s %9s %7s' % ('kHz', 'nRX', 'A0_des', 'v_med', 'v_lo', 'v_hi', 'pk_max'))
summary = []
for fk, job in JOBS:
    rows = extract(job)
    vs = np.array([r[3] for r in rows if np.isfinite(r[3]) and r[3] > 0])
    pks = np.array([r[1] for r in rows])
    # use far-field RX (dist>120mm) for cleaner group-velocity estimate
    far = np.array([r[3] for r in rows if np.isfinite(r[3]) and r[0] > 120])
    vmed = np.median(far) if far.size else np.median(vs)
    summary.append((fk, A0_DESIGN[fk], vmed, vs.min(), vs.max(), pks.max(), len(rows)))
    print('%5d %6d %9.0f %9.0f %9.0f %9.0f %7.1e'
          % (fk, len(rows), A0_DESIGN[fk], vmed, vs.min(), vs.max(), pks.max()))

print('\n--- dispersion tracking (FEM far-field median vs A0 design) ---')
for fk, des, vmed, vlo, vhi, pk, n in summary:
    err = 100.0 * (vmed - des) / des
    print('%3d kHz: design %4.0f  FEM %4.0f  (%+5.1f%%)' % (fk, des, vmed, err))
print('\nExpect FEM median to RISE with freq (A0 dispersive) and sit near design +/-15%.')
np.savez('rx_batch_summary.npz', summary=np.array([(s[0], s[1], s[2]) for s in summary]))
print('saved rx_<job>.npz (per-freq waveforms) + rx_batch_summary.npz')
