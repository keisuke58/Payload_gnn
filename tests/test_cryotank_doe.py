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
                         "pressure_MPa": 0.3, "theta_deg": 45.0, "z_center": 2000, "radius": 80})
    assert rt["material"]["E_MPa"] == m.MAT_ALLI["rt"]["E_MPa"]
    assert rt["defect"]["type"] == "weld_flaw"
