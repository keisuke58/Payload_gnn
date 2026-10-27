"""Smoke tests for the hydrogen-tank DOE generator + FEM scaffold (no Abaqus)."""

import importlib


def test_cryotank_doe_generate_one():
    m = importlib.import_module("generate_cryotank_doe")
    doe = m.generate(1, seed=42)
    assert doe["metadata"]["n_samples"] == 1
    assert len(doe["samples"]) == 1
    dp = doe["samples"][0]["defect_params"]
    assert dp["defect_type"] in (["healthy"] + m.DEFECT_TYPES)
    assert dp["temperature_K"] in m.TEMPERATURE_K


def test_cryotank_doe_batch_and_bounds():
    m = importlib.import_module("generate_cryotank_doe")
    doe = m.generate(50, seed=7)
    assert len(doe["samples"]) == 50
    for s in doe["samples"]:
        dp = s["defect_params"]
        assert m.THETA_RANGE[0] <= dp["theta_deg"] <= m.THETA_RANGE[1] or dp["defect_type"] == "weld_flaw"
        assert m.Z_RANGE[0] <= dp["z_center"] <= m.Z_RANGE[1]
        assert dp["radius"] > 0


def test_cryotank_healthy_only():
    m = importlib.import_module("generate_cryotank_doe")
    doe = m.generate(3, healthy_only=True)
    assert all(s["defect_params"]["defect_type"] == "healthy" for s in doe["samples"])


def test_cryotank_dataset_dry_run_plan():
    m = importlib.import_module("generate_cryotank_dataset")
    # cryogenic operating point → E should be scaled up vs RT
    plan = m.resolve_plan({"defect_type": "healthy", "temperature_K": 20.0, "pressure_MPa": 0.3})
    assert plan["material"]["E_MPa"] > m.MAT_ALLI["rt"]["E_MPa"]
    assert plan["loading"]["thermal"]["delta_K"] < 0  # cooling from RT reference
    rt = m.resolve_plan({"defect_type": "weld_flaw", "temperature_K": 293.0,
                         "pressure_MPa": 0.3, "theta_deg": 15.0, "z_center": 2000, "radius": 80})
    assert rt["material"]["E_MPa"] == m.MAT_ALLI["rt"]["E_MPa"]
    assert rt["defect"]["type"] == "weld_flaw"


def test_cryotank_doe_patches_fit_model_sector():
    doe_mod = importlib.import_module("generate_cryotank_doe")
    ds = importlib.import_module("generate_cryotank_dataset")
    assert doe_mod.SECTOR_DEG == ds.SECTOR_DEG
    doe = doe_mod.generate(300, seed=11)
    for s in doe["samples"]:
        plan = ds.resolve_plan(s["defect_params"])
        th_lo, th_hi = plan["defect"]["patch"]["theta_deg"]
        z_lo, z_hi = plan["defect"]["patch"]["z_mm"]
        # patch is a real, non-degenerate box strictly inside the 30 deg sector/barrel
        assert 0.0 < th_lo < th_hi < ds.SECTOR_DEG
        assert 0.0 < z_lo < z_hi < ds.H_BARREL
        # the theta clamp never kicks in, i.e. the full defect circle is modelled
        half = (th_hi - th_lo) / 2.0
        assert abs(half - __import__("math").degrees(s["defect_params"]["radius"] / ds.RADIUS)) < 1e-3


def test_cryotank_patch_rejects_out_of_sector():
    import pytest
    ds = importlib.import_module("generate_cryotank_dataset")
    with pytest.raises(ValueError):
        ds.defect_patch(45.0, 2000.0, 80.0)   # old DOE range produced theta up to 85 deg
    with pytest.raises(ValueError):
        ds.defect_patch(15.0, 5000.0, 80.0)   # beyond the 4000 mm barrel


def test_cryotank_end_cap_and_bottom_bc():
    import pytest
    m = importlib.import_module("generate_cryotank_dataset")
    p = 0.4
    plan = m.resolve_plan({"defect_type": "healthy", "temperature_K": 20.0, "pressure_MPa": p})
    # closed-end by default: N_z = pR/2, and the default bottom is the skirt attachment
    assert plan["loading"]["closed_end"] is True
    assert abs(plan["loading"]["end_cap_thrust_N_per_mm"] - p * m.RADIUS / 2.0) < 1e-9
    assert plan["boundary"]["bottom"] == "skirt"
    open_plan = m.resolve_plan({"defect_type": "healthy", "pressure_MPa": p,
                                "closed_end": False, "bottom_bc": "axial"})
    assert open_plan["loading"]["end_cap_thrust_N_per_mm"] == 0.0
    assert open_plan["boundary"]["bottom"] == "axial"
    with pytest.raises(ValueError):
        m.resolve_plan({"defect_type": "healthy", "bottom_bc": "clamped"})
