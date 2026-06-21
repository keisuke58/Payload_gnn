# abq2024 python extract_rx.py   (run with Abaqus python: has odbAccess + numpy)
# Pull RX-node displacement histories from the COPV GW odb, save npz, print a
# propagation sanity check (first-arrival time-of-flight vs A0 group velocity).
from odbAccess import openOdb
import numpy as np, math

odb = openOdb('COPV_GW.odb', readOnly=True)
step = odb.steps['gw']
asm = odb.rootAssembly

# TX node coords (transmitter) from the TX set
tx_set = asm.nodeSets['TX']
txn = tx_set.nodes[0][0]
txc = np.array(txn.coordinates)

series = {}    # label -> (t, u1,u2,u3)
coords = {}
for rname, region in step.historyRegions.items():
    outs = region.historyOutputs
    if not any(k in outs for k in ('U1', 'U2', 'U3')):
        continue
    t = None; comp = {}
    for c in ('U1', 'U2', 'U3'):
        if c in outs:
            d = np.array(outs[c].data)       # (npts,2): time,value
            t = d[:, 0]; comp[c] = d[:, 1]
    series[rname] = (t, comp)
    # node coords: region.point.node
    try:
        nlabel = region.point.node.label
        ncoord = region.point.node.coordinates
        coords[rname] = np.array(ncoord)
    except Exception:
        coords[rname] = None

print('RX history regions: %d' % len(series))
# radial displacement magnitude per RX, first-arrival (3% of own peak), TOF check
cg_A0 = 2900.0   # m/s representative A0 group velocity near 100 kHz [U]
rows = []
for rname, (t, comp) in sorted(series.items()):
    u = np.zeros_like(t)
    for c in comp:
        u = u + comp[c] ** 2
    u = np.sqrt(u)               # |U| magnitude
    pk = u.max()
    c = coords[rname]
    dist = np.nan
    if c is not None:
        dist = np.sqrt(((c - txc) ** 2).sum())   # mm straight-line (chord)
    thr = 0.03 * pk if pk > 0 else 0
    ia = np.argmax(u > thr) if pk > 0 else -1
    ta = t[ia] if ia >= 0 else np.nan
    v_app = (dist * 1e-3) / ta if (ta and ta > 0 and np.isfinite(dist)) else np.nan
    rows.append((rname, dist, pk, ta, v_app))

rows.sort(key=lambda r: (np.inf if np.isnan(r[1]) else r[1]))
print('%-22s %8s %11s %10s %9s' % ('RXregion', 'dist_mm', 'peak|U|', 't_arr_s', 'v_app m/s'))
for rname, dist, pk, ta, v in rows:
    print('%-22s %8.1f %11.3e %10.3e %9.0f' % (rname, dist, pk, ta, (v if np.isfinite(v) else -1)))

# save
np.savez('rx_history.npz',
         **{('%s_%s' % (r, c)): series[r][1][c] for r in series for c in series[r][1]},
         **{('%s_t' % r): series[r][0] for r in series},
         tx_coord=txc)
peaks = [r[2] for r in rows]
print('\nSUMMARY: peak|U| range %.2e .. %.2e mm; nonzero RX = %d/%d'
      % (min(peaks), max(peaks), sum(1 for p in peaks if p > 1e-12), len(peaks)))
print('apparent first-arrival speeds should sit between A0(~2900) and S0(~5000) m/s if a wave truly propagates.')
odb.close()
