# abaqus python gen_damage_library_inp.py   (Abaqus Python environment: has odbAccess + numpy)
"""Generate N modified INP files for COPV guided-wave damage training library.

Strategy: Add a concentrated MASS element at the mesh node nearest to each
target damage position. Mass loading creates a local impedance mismatch that
scatters the A0 wave — matching the BAM "steel block" reversible-damage test.

Requires the healthy baseline ODB to be available so that node coordinates
can be extracted. Run AFTER the healthy baseline job completes.

Usage (on frontale, after module abaqus/2024 loaded):
  abaqus python gen_damage_library_inp.py \\
      --odb COPV_GW_180k.odb \\
      --inp COPV_GW_180k.inp \\
      --n_damage 80 --mass_tonne 1e-4 --seed 0
  -> creates damage_lib/COPV_GW_180k_d{000..079}.inp
     and   damage_lib/damage_positions.json
"""
from odbAccess import openOdb
import numpy as np, json, os, argparse, math

p = argparse.ArgumentParser()
p.add_argument('--odb',       default='COPV_GW_180k.odb')
p.add_argument('--inp',       default='COPV_GW_180k.inp')
p.add_argument('--n_damage',  type=int,   default=80)
p.add_argument('--mass_tonne',type=float, default=1e-4)   # ~100g; BAM block ~200g
p.add_argument('--seed',      type=int,   default=0)
p.add_argument('--outdir',    default='damage_lib')
a, _ = p.parse_known_args()

os.makedirs(a.outdir, exist_ok=True)

# ─── 1. Extract Part-level node coordinates from ODB ──────────────────────────
print('Loading ODB: %s' % a.odb)
odb     = openOdb(a.odb, readOnly=True)
part    = list(odb.parts.values())[0]
nodes   = part.nodes
node_labels  = np.array([n.label for n in nodes])
node_coords  = np.array([n.coordinates for n in nodes])   # (N, 3)  x,y,z mm
odb.close()
print('Part nodes: %d' % len(node_labels))

# Max existing element label (to create a new unique MASS element label)
# Read from INP rather than ODB (ODB may not expose element labels easily)
max_elem = 0
with open(a.inp) as f:
    in_elem = False
    for line in f:
        ls = line.strip()
        if ls.startswith('*Element'):
            in_elem = True; continue
        if ls.startswith('*') and in_elem:
            in_elem = False
        if in_elem:
            try:
                lbl = int(ls.split(',')[0].strip())
                if lbl > max_elem:
                    max_elem = lbl
            except ValueError:
                pass
print('Max element label in INP: %d' % max_elem)

# ─── 2. Build search tree ─────────────────────────────────────────────────────
# The FEM is a 90° barrel sector at R≈176mm, z in [0, 500mm].
# All surface nodes are on the outer shell.
# Abaqus Python has no scipy — brute-force nearest-neighbour (80 queries × 348k nodes ~OK)
def _nn_query(coords, targets):
    idxs, dists = [], []
    for t in targets:
        d = np.sqrt(((coords - t) ** 2).sum(axis=1))
        k = int(d.argmin())
        idxs.append(k); dists.append(float(d[k]))
    return np.array(dists), np.array(idxs)

# ─── 3. Sample N damage positions within the FEM domain ───────────────────────
# Unrolled sector: circ ∈ [-R*π/4, R*π/4] ≈ [-138, 138] mm; axial ∈ [20, 480] mm
# (small margin from boundaries)
R    = 176.0
circ_min, circ_max  = -R * math.pi / 4 + 10, R * math.pi / 4 - 10
axial_min, axial_max = 20.0, 480.0

rng = np.random.default_rng(a.seed)
circ_pos  = rng.uniform(circ_min, circ_max, a.n_damage)
axial_pos = rng.uniform(axial_min, axial_max, a.n_damage)

# Convert unrolled (circ, axial) to 3D (x, y, z) on the cylinder surface
theta = circ_pos / R   # radians from θ=0
targets = np.column_stack([R * np.cos(theta), R * np.sin(theta), axial_pos])

# ─── 4. Find nearest Part-level node for each target ─────────────────────────
dists, idxs = _nn_query(node_coords, targets)
nearest_labels = node_labels[idxs]
actual_coords  = node_coords[idxs]

positions_json = []
for i in range(a.n_damage):
    positions_json.append({
        'id':           i,
        'circ_mm':      float(circ_pos[i]),
        'axial_mm':     float(axial_pos[i]),
        'theta_deg':    float(math.degrees(theta[i])),
        'node_label':   int(nearest_labels[i]),
        'node_x':       float(actual_coords[i, 0]),
        'node_y':       float(actual_coords[i, 1]),
        'node_z':       float(actual_coords[i, 2]),
        'snap_dist_mm': float(dists[i]),
    })

json.dump(positions_json, open(os.path.join(a.outdir, 'damage_positions.json'), 'w'), indent=2)
print('Wrote damage_positions.json (%d entries)' % a.n_damage)

# ─── 5. Read healthy INP as list of lines ────────────────────────────────────
with open(a.inp) as f:
    base_lines = f.readlines()

# Find insertion point: just before *End Part
end_part_idx = None
for k, ln in enumerate(base_lines):
    if ln.strip().lower().startswith('*end part'):
        end_part_idx = k; break
if end_part_idx is None:
    raise RuntimeError('*End Part not found in INP')

# Find insertion point for job name: line starting with *Heading or ** Job name
job_name_idx = None
for k, ln in enumerate(base_lines):
    if '** Job name:' in ln:
        job_name_idx = k; break

# ─── 6. Generate one INP per damage position ─────────────────────────────────
for i, pos in enumerate(positions_json):
    tag       = 'COPV_GW_180k_d%03d' % i
    node_lbl  = pos['node_label']
    elem_lbl  = max_elem + 1 + i   # unique element label

    # MASS element block to insert before *End Part
    mass_block = [
        '**\n',
        '** Damage perturbation: concentrated mass at node %d (d=%.1fmm)\n'
            % (node_lbl, pos['snap_dist_mm']),
        '*Element, type=MASS, elset=DAMAGE_MASS\n',
        '%d, %d\n' % (elem_lbl, node_lbl),
        '*Mass, elset=DAMAGE_MASS\n',
        '%.6e,\n' % a.mass_tonne,
    ]

    new_lines = (base_lines[:end_part_idx]
                 + mass_block
                 + base_lines[end_part_idx:])

    # Patch job name in header
    if job_name_idx is not None:
        new_lines[job_name_idx] = '** Job name: %s Model name: Model-1\n' % tag

    out_path = os.path.join(a.outdir, tag + '.inp')
    with open(out_path, 'w') as f:
        f.writelines(new_lines)

print('Generated %d INP files in %s/' % (a.n_damage, a.outdir))
print('Run: bash submit_damage_library.sh')
