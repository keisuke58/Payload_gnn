"""Run build_model() against a geometric fake of the Abaqus CAE API (no Abaqus needed).

This does NOT validate Abaqus behaviour; it executes our own control flow and the
geometric selection logic (defect patch, symmetry cut planes, tie faces) on synthetic
faces/edges/cells laid out like the real revolved parts, for both revolve directions.
"""

import importlib
import math
import sys
import types
from unittest import mock

import pytest

CONSTANTS = [
    "NO_IDEALIZATION", "DEFAULT", "SIMPSON", "THREE_D", "DEFORMABLE_BODY", "OFF", "ON",
    "XZPLANE", "STANDALONE", "TRACTION", "ENERGY", "BK", "TRACTION_SEPARATION", "SPECIFY",
    "HEX", "SWEEP", "COH3D8", "COH3D6", "STANDARD", "TET", "FREE", "C3D10", "UNIFORM",
    "GENERAL", "CARTESIAN", "CYLINDRICAL", "SPECIFIED", "ANALYSIS", "ODB",
]


class Ent:
    def __init__(self, idx, pt):
        self.index = idx
        self.pointOn = (pt,)


class Seq(list):
    def __getitem__(self, k):
        r = list.__getitem__(self, k)
        return Seq(r) if isinstance(k, slice) else r

    def __add__(self, other):
        return Seq(list(self) + list(other))

    def getByBoundingBox(self, xMin=-1e30, yMin=-1e30, zMin=-1e30,
                         xMax=1e30, yMax=1e30, zMax=1e30):
        return Seq(e for e in self if xMin <= e.pointOn[0][0] <= xMax
                   and yMin <= e.pointOn[0][1] <= yMax and zMin <= e.pointOn[0][2] <= zMax)


def _seq(points):
    return Seq(Ent(i, p) for i, p in enumerate(points))


class Sketch:
    def __init__(self):
        self.lines = []

    def Line(self, point1, point2):
        self.lines.append((point1, point2))

    def __getattr__(self, name):          # setPrimaryObject, ConstructionLine, ...
        return lambda *a, **k: None


class Part:
    SGN = 1.0

    def __init__(self, name):
        self.name = name
        self.faces = self.edges = self.cells = Seq()
        self.sets = {}
        self.section_calls = []
        self.radial_planes = []

    def _pt(self, r, th_deg, y):
        th = math.radians(th_deg)
        return (r * math.cos(th), y, self.SGN * r * math.sin(th))

    def _revolve(self, sketch, angle, solid):
        xs = [p[0] for ln in sketch.lines for p in ln]
        ys = [p[1] for ln in sketch.lines for p in ln]
        h = max(ys)
        ths = [angle * (i + 0.5) / 60 for i in range(60)]   # 0.5 deg grid
        yy = [h * (j + 0.5) / 40 for j in range(40)]         # 100 mm grid
        if solid:
            r_in, r_out = min(xs), max(xs)
            rm = 0.5 * (r_in + r_out)
            self.cells = _seq([self._pt(rm, t, y) for t in ths for y in yy])
            faces = [self._pt(r, t, y) for r in (r_in, r_out) for t in ths for y in yy]
            faces += [self._pt(rm, a, y) for a in (0.0, angle) for y in yy]  # cut faces
            self.faces = _seq(faces)
        else:
            r = xs[0]
            self.faces = _seq([self._pt(r, t, y) for t in ths for y in yy])
            edges = [self._pt(r, a, y) for a in (0.0, angle) for y in yy]    # cut edges
            edges += [self._pt(r, t, y) for t in ths for y in (0.0, h)]      # bottom/top
            self.edges = _seq(edges)

    def BaseShellRevolve(self, sketch, angle, flipRevolveDirection):
        self._revolve(sketch, angle, solid=False)

    def BaseSolidRevolve(self, sketch, angle, flipRevolveDirection):
        self._revolve(sketch, angle, solid=True)

    def DatumPlaneByThreePoints(self, point1, point2, point3):
        self.radial_planes.append(point3)
        return mock.MagicMock()

    def Set(self, name, faces=None, cells=None, edges=None):
        s = Seq(faces or cells or edges or [])
        self.sets[name] = s
        return s

    def SectionAssignment(self, region, sectionName):
        self.section_calls.append((sectionName, len(region)))

    def __getattr__(self, name):          # datum planes, partitions, ...
        return mock.MagicMock()


class Instance:
    def __init__(self, name, part):
        self.name = name
        self.faces, self.edges, self.cells = part.faces, part.edges, part.cells
        self.sets = part.sets
        self.elements = self.nodes = [0] * 10


class Assembly(mock.MagicMock):
    def Instance(self, name, part, dependent):
        return Instance(name, part)

    def Set(self, name, **kw):
        (v,) = kw.values()
        self.sets_made[name] = len(v)
        return v


@pytest.fixture
def fake_abaqus(monkeypatch):
    consts = types.ModuleType("abaqusConstants")
    for c in CONSTANTS:
        setattr(consts, c, c)
    model = mock.MagicMock()
    parts = {}

    def make_part(name, dimensionality, type):
        parts[name] = Part(name)
        return parts[name]

    model.Part.side_effect = make_part
    model.ConstrainedSketch.side_effect = lambda **k: Sketch()
    asm = Assembly()
    asm.sets_made = {}
    model.rootAssembly = asm
    abaqus = types.ModuleType("abaqus")
    abaqus.mdb = mock.MagicMock(models={"Model-1": model})
    mesh = types.ModuleType("mesh")
    mesh.ElemType = mock.MagicMock()
    drv = types.ModuleType("driverUtils")
    drv.executeOnCaeStartup = lambda: None
    for name, mod in (("abaqus", abaqus), ("abaqusConstants", consts),
                      ("caeModules", types.ModuleType("caeModules")),
                      ("driverUtils", drv), ("mesh", mesh)):
        monkeypatch.setitem(sys.modules, name, mod)
    ds = importlib.reload(importlib.import_module("generate_cryotank_dataset"))
    assert ds._HAVE_ABAQUS
    yield ds, model, parts, asm
    for name in ("abaqus", "abaqusConstants", "caeModules", "driverUtils", "mesh"):
        sys.modules.pop(name, None)
    importlib.reload(ds)


@pytest.mark.parametrize("sgn", [1.0, -1.0])
@pytest.mark.parametrize("params", [
    {"defect_type": "healthy", "temperature_K": 20.0, "pressure_MPa": 0.37},
    {"defect_type": "weld_flaw", "temperature_K": 20.0, "pressure_MPa": 0.37,
     "theta_deg": 15.0, "z_center": 1500.0, "radius": 120.0},
    {"defect_type": "insulation_debond", "temperature_K": 20.0, "pressure_MPa": 0.37,
     "theta_deg": 9.0, "z_center": 2600.0, "radius": 200.0},
])
def test_build_model_runs_and_selects(fake_abaqus, capsys, params, sgn, tmp_path, monkeypatch):
    ds, model, parts, asm = fake_abaqus
    Part.SGN = sgn
    monkeypatch.chdir(tmp_path)
    ds.build_model("Smoke", params)
    out = capsys.readouterr().out
    assert "WARNING" not in out, out

    dtype = params["defect_type"]
    shells = ["Tank-1"] + (["Foam-1"] if dtype == "insulation_debond" else [])
    solids = ["Bond-1"] if dtype == "insulation_debond" else []
    # symmetry BCs: both cut planes on every instance; shells have 40 cut edges per plane
    for inst in shells + solids:
        for tag in ("T0", "Ta"):
            assert asm.sets_made["BC_Sym_%s_%s" % (tag, inst)] == 40
    # loads: pressure + closed-end thrust pR/2
    p = params["pressure_MPa"]
    assert model.Pressure.call_args.kwargs["magnitude"] == p
    assert math.isclose(model.ShellEdgeLoad.call_args.kwargs["magnitude"], p * ds.RADIUS / 2)
    assert asm.sets_made["BC_Bottom"] == 60
    # theta=sector symmetry CSYS and radial patch planes follow the revolve direction
    csys = [c.kwargs for c in asm.DatumCsysByThreePoints.call_args_list
            if c.kwargs.get("name") == "CSYS-SymTheta"]
    assert len(csys) == 1 and math.copysign(1.0, csys[0]["point1"][2]) == sgn
    for part in parts.values():
        for p3 in part.radial_planes:
            assert math.copysign(1.0, p3[2]) == sgn

    plan = ds.resolve_plan(params)
    if dtype == "weld_flaw":
        sel = parts["Tank"].sets["Set-Defect"]
        assert sel and ("Sec-Defect", len(sel)) in parts["Tank"].section_calls
    if dtype == "insulation_debond":
        sel = parts["Bond"].sets["Set-Debond"]
        (th_lo, th_hi), (z_lo, z_hi) = plan["defect"]["patch"].values()
        for c in sel:   # every selected cell lies inside the patch
            x, y, z = c.pointOn[0]
            assert z_lo < y < z_hi and th_lo < math.degrees(abs(math.atan2(z, x))) < th_hi
        assert ("Sec-Bond-Debonded", len(sel)) in parts["Bond"].section_calls
        assert model.Tie.call_count == 2
        assert model.CohesiveSection.call_count == 2
        assert "Set-Defect" not in parts["Tank"].sets   # wall stays healthy for debond
