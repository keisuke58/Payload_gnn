# abq2024 cae noGUI=generate_copv_gw.py -- --freq 100e3 --sector 90 --length 500 --no_run
# COPV guided-wave model (BAM Type-IV, OD 352mm, 700 bar).
# Barrel SECTOR + S4R orthotropic composite shell, Hann tone-burst PZT excitation,
# history output at receiver nodes.
#
# Layup [S] (inner -> outer, filament-wound Type-IV COPV; ply thicknesses [U]):
#   CF_90(1mm) / CF+54(1.5mm) / CF-54(1.5mm) / CF_90(1mm)
#   CF+54(1.5mm) / CF-54(1.5mm) / CF_90(1mm) / GFRP_0(1mm)   total = 10 mm
# CF  : T700M21      E1=125.5 E2=8.7 G12=4.135 GPa, nu12=0.37 nu23=0.45  [S]
# GFRP: TVR380M12R   E1=46.4  E2=14.9 G12=5.23 GPa, nu12=0.27 nu23=0.30  [S]
# Source: Dispersion-Calculator/DC_MATLAB_Code/Materials/MaterialList_TransverselyIsotropic.txt
#
# Dispersion table (CP[]): CLT-effective axial approximation [S].
# After any layup change, rerun dispersion_copv.py and update CP[] from its printed table.
# Conservative 70% of original homogenized values used until recomputed (gives finer mesh).
from abaqus import *
from abaqusConstants import *
import section
import mesh, regionToolset, math, argparse, sys, os

p = argparse.ArgumentParser()
p.add_argument('--job', default='COPV_GW')
p.add_argument('--freq', type=float, default=100e3)   # excitation (Hz); real set 60-300kHz
p.add_argument('--sector', type=float, default=90.0)  # barrel arc [deg]
p.add_argument('--length', type=float, default=500.0) # barrel axial length [mm]
p.add_argument('--ncyc', type=int, default=5)         # tone-burst cycles (Hann)
p.add_argument('--epw', type=float, default=18.0)     # elements per A0 wavelength
p.add_argument('--no_run', action='store_true')
a, _ = p.parse_known_args()

# --- geometry (BAM COPV: OD 352 mm) ---
R = 352.0 / 2.0
arc = math.radians(a.sector)
Lz = a.length

# --- A0 phase-velocity table for mesh sizing [S] ---
# Computed by dispersion_copv.py (CLT A-matrix -> Ex=17.3 GPa, nxy=0.245, rho=1594 kg/m3).
# Old homogenized (E=50 GPa): 2145->3109 m/s.  New CLT: 1476->1895 m/s (-30 to -40%).
# Finer mesh results (see lam/20 = 0.316 mm at 300 kHz). Rerun dispersion_copv.py after
# any layup change and paste the printed A0 cp column here.
fk = a.freq / 1e3
import bisect
F  = [60,    120,   180,   260,   300  ]
CP = [1476., 1727., 1826., 1881., 1895.]  # [S] CLT Ex-eff. dispersion_copv.py 2026-06-21
if fk <= F[0]: cpA0 = CP[0]
elif fk >= F[-1]: cpA0 = CP[-1]
else:
    i = bisect.bisect(F, fk) - 1
    t_frac = (fk - F[i]) / (F[i + 1] - F[i])
    cpA0 = CP[i] + t_frac * (CP[i + 1] - CP[i])
lamA0 = cpA0 / a.freq * 1e3            # mm
seed  = lamA0 / a.epw                  # mm element size
print('freq=%.0f kHz  A0 cp=%.0f m/s  lamA0=%.2f mm  -> seed=%.3f mm' % (fk, cpA0, lamA0, seed))

m = mdb.models['Model-1']
sh = m.ConstrainedSketch(name='sec', sheetSize=2 * R)
x0, y0 = R * math.cos(-arc / 2), R * math.sin(-arc / 2)
x1, y1 = R * math.cos(arc / 2), R * math.sin(arc / 2)
sh.ArcByCenterEnds(center=(0, 0), point1=(x0, y0), point2=(x1, y1), direction=COUNTERCLOCKWISE)
prt = m.Part(name='barrel', dimensionality=THREE_D, type=DEFORMABLE_BODY)
prt.BaseShellExtrude(sketch=sh, depth=Lz)

# --- CFRP material: T700M21, transversely isotropic [S] ---
# Units throughout: N/mm² (MPa) and tonne/mm³  (Abaqus mm-tonne-s system)
# Transverse isotropy: E3=E2, G13=G12, nu13=nu12
G23_CF = 8.7e3 / (2.0 * (1.0 + 0.45))   # 3000 MPa
cfrp = m.Material(name='CFRP')
cfrp.Density(table=((1.571e-9,),))        # 1571 kg/m3 -> tonne/mm3
cfrp.Elastic(
    type=ENGINEERING_CONSTANTS,
    table=((125.5e3, 8.7e3,   8.7e3,      # E1,  E2,  E3
            0.37,   0.37,    0.45,         # nu12, nu13, nu23
            4.135e3, 4.135e3, G23_CF),)   # G12, G13, G23
)

# --- GFRP material: TVR380M12R, transversely isotropic [S] ---
G23_GF = 14.92e3 / (2.0 * (1.0 + 0.30)) # 5738 MPa
gfrp = m.Material(name='GFRP')
gfrp.Density(table=((1.8e-9,),))          # 1800 kg/m3 -> tonne/mm3
gfrp.Elastic(
    type=ENGINEERING_CONSTANTS,
    table=((46.43e3, 14.92e3, 14.92e3,    # E1,  E2,  E3
            0.27,   0.27,    0.30,         # nu12, nu13, nu23
            5.233e3, 5.233e3, G23_GF),)   # G12, G13, G23
)

# --- Composite shell section [S] ---
# Winding rationale:
#   +-54 deg helical: near-optimal for internal pressure (netting theory alpha_opt ~54.7 deg)
#   90 deg hoop: supplement hoop load capacity between helical bands
#   0 deg GFRP outer: impact / UV protection layer (lower E1 than CF -> minor stiffness effect)
# Orientations relative to shell local-1 axis (= cylinder axial direction Z).
#   90 deg -> fiber in hoop (circumferential) direction
#  +-54 deg -> fiber at +-54 deg to axial
#    0 deg -> fiber in axial direction (GFRP outer)
LAYUP_DEF = (
    # (material_name, thickness_mm, orient_deg)   inner -> outer
    ('CFRP',  1.0,  90.0),   # CF hoop inner
    ('CFRP',  1.5,  54.0),   # CF helical +
    ('CFRP',  1.5, -54.0),   # CF helical -
    ('CFRP',  1.0,  90.0),   # CF hoop mid-inner
    ('CFRP',  1.5,  54.0),   # CF helical +
    ('CFRP',  1.5, -54.0),   # CF helical -
    ('CFRP',  1.0,  90.0),   # CF hoop outer
    ('GFRP',  1.0,   0.0),   # GFRP protective outer (axial fiber)
)  # total wall = 10.0 mm

layup_layers = tuple(
    section.SectionLayer(material=mat, thickness=tk, orientAngle=ang, numIntPts=3)
    for mat, tk, ang in LAYUP_DEF
)
m.CompositeShellSection(
    name='wall',
    preIntegrate=OFF,
    idealization=NO_IDEALIZATION,
    symmetric=False,
    thicknessType=UNIFORM,
    poissonDefinition=DEFAULT,
    layup=layup_layers,
)
reg = regionToolset.Region(faces=prt.faces)
prt.SectionAssignment(region=reg, sectionName='wall')

prt.seedPart(size=seed, deviationFactor=0.1)
prt.setElementType(regions=reg, elemTypes=(mesh.ElemType(elemCode=S4R, elemLibrary=EXPLICIT),))
prt.generateMesh()
ne = len(prt.elements); nn = len(prt.nodes)
print('MESH: %d elements, %d nodes (seed %.3f mm)' % (ne, nn, seed))

asm = m.rootAssembly
inst = asm.Instance(name='b1', part=prt, dependent=ON)

# PZT nodes: 5x5 grid on the sector
def node_at(theta, z):
    x, y = R * math.cos(theta), R * math.sin(theta)
    return inst.nodes.getClosest((x, y, z))
ths = [(-arc / 2 + arc * i / 4.0) for i in range(5)]
zs = [Lz * (i + 0.5) / 5.0 for i in range(5)]
pzt = [node_at(th, z) for z in zs for th in ths]
tx = pzt[12]  # center = transmitter
asm.Set(name='TX', nodes=mesh.MeshNodeArray([tx]))
asm.Set(name='RX', nodes=mesh.MeshNodeArray([n for n in pzt if n.label != tx.label]))

# Explicit dynamic step; total = burst + ~3 transit windows
CG = 3.0e6   # approximate group velocity mm/s
T  = a.ncyc / a.freq + 3.0 * Lz / CG
m.ExplicitDynamicsStep(name='gw', previous='Initial', timePeriod=T,
                       nlgeom=OFF, improvedDtMethod=ON)

# Hann-windowed tone-burst at TX radial DOF
N  = 400
tb = a.ncyc / a.freq
amp = []
for i in range(N + 1):
    ti = tb * i / N
    w = 0.5 * (1 - math.cos(2 * math.pi * a.freq * ti / a.ncyc)) if ti <= tb else 0.0
    amp.append((ti, w * math.sin(2 * math.pi * a.freq * ti)))
m.TabularAmplitude(name='burst', data=tuple(amp))
thx = math.atan2(tx.coordinates[1], tx.coordinates[0])
m.ConcentratedForce(name='exc', createStepName='gw', region=asm.sets['TX'],
                    cf1=math.cos(thx), cf2=math.sin(thx), amplitude='burst')

# Diagnostic log
_lg = open('/home/nishioka/Payload2026/copv_gw/gen_probe.txt', 'w')
_lg.write('type(m)=%s\n' % str(type(m)))
_lg.write('has fieldOutputRequests=%s\n' % hasattr(m, 'fieldOutputRequests'))
_lg.write('has historyOutputRequests=%s\n' % hasattr(m, 'historyOutputRequests'))
try:
    _lg.write('steps=%s\n' % str(list(m.steps.keys())))
except Exception as e:
    _lg.write('steps ERR %s\n' % str(e)[:80])
try:
    _lg.write('fieldReqs=%s\n' % str(list(m.fieldOutputRequests.keys())))
except Exception as e:
    _lg.write('fieldReqs ERR %s\n' % str(e)[:80])
_lg.write('elements=%d nodes=%d seed=%.3f\n' % (ne, nn, seed))
outattrs = [x for x in dir(m) if 'utput' in x or 'equest' in x]
_lg.write('output-related model attrs=%s\n' % str(outattrs))
_lg.flush(); _lg.close()

dt_out = 1.0 / 2.976e6   # sample at real 2.976 MHz

mdb.Job(name=a.job, model='Model-1', type=ANALYSIS, explicitPrecision=DOUBLE,
        nodalOutputPrecision=FULL)
mdb.jobs[a.job].writeInput(consistencyChecking=OFF)
print('WROTE %s.inp  (elements=%d)  total_time=%.3e s' % (a.job, ne, T))
if not a.no_run:
    mdb.jobs[a.job].submit(); mdb.jobs[a.job].waitForCompletion()
