#!/usr/bin/env python3
"""Audit generated FEM/graph CSV datasets for SHM dataset reliability.

This script is intentionally dependency-light (standard library only) so it can
run in minimal CI environments without Abaqus, pandas, torch, or PyG.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REQUIRED_NODE_COLUMNS = {"node_id", "x", "y", "z", "defect_label"}
PHYSICS_COLUMNS = [
    "ux",
    "uy",
    "uz",
    "u_mag",
    "temp",
    "s11",
    "s22",
    "s12",
    "smises",
    "dspss",
    "thermal_smises",
    "le11",
    "le22",
    "le12",
]
DISPLACEMENT_COLUMNS = ["ux", "uy", "uz", "u_mag"]
STRESS_COLUMNS = ["s11", "s22", "s12", "smises", "dspss", "thermal_smises"]
ELEMENT_COLUMNS = {"elem_id", "n1", "n2", "n3"}


def _safe_float(value: str) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _discover_sample_dirs(root: Path) -> List[Path]:
    sample_dirs: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        path = Path(dirpath)
        if "nodes.csv" in filenames:
            sample_dirs.append(path)
            dirnames[:] = []
    return sorted(sample_dirs)


def _read_metadata(sample_dir: Path) -> Dict[str, str]:
    path = sample_dir / "metadata.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if rows and rows[0] == ["key", "value"]:
        return {row[0]: row[1] for row in rows[1:] if len(row) >= 2}
    return {}


def _count_elements(sample_dir: Path) -> Tuple[int, List[str], List[str]]:
    path = sample_dir / "elements.csv"
    if not path.exists():
        return 0, [], ["missing elements.csv"]
    warnings: List[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = sorted(ELEMENT_COLUMNS - set(columns))
        if missing:
            warnings.append("elements.csv missing columns: " + ",".join(missing))
        count = sum(1 for _ in reader)
    return count, columns, warnings


def _audit_nodes(sample_dir: Path) -> Dict[str, object]:
    path = sample_dir / "nodes.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing_required = sorted(REQUIRED_NODE_COLUMNS - set(columns))
        present_physics = [c for c in PHYSICS_COLUMNS if c in columns]
        present_displacement = [c for c in DISPLACEMENT_COLUMNS if c in columns]
        present_stress = [c for c in STRESS_COLUMNS if c in columns]

        n_nodes = 0
        labels: Counter[str] = Counter()
        nonzero_counts: Counter[str] = Counter()
        maxima: Dict[str, float] = {}
        minima: Dict[str, float] = {}
        coordinate_bounds = {
            "x": [math.inf, -math.inf],
            "y": [math.inf, -math.inf],
            "z": [math.inf, -math.inf],
        }

        for row in reader:
            n_nodes += 1
            if "defect_label" in row:
                labels[row["defect_label"]] += 1
            for axis in ("x", "y", "z"):
                if axis in row:
                    value = _safe_float(row[axis])
                    coordinate_bounds[axis][0] = min(coordinate_bounds[axis][0], value)
                    coordinate_bounds[axis][1] = max(coordinate_bounds[axis][1], value)
            for col in present_physics:
                value = _safe_float(row[col])
                maxima[col] = max(maxima.get(col, abs(value)), abs(value))
                minima[col] = min(minima.get(col, value), value)
                if abs(value) > 1e-12:
                    nonzero_counts[col] += 1

    displacement_nonzero = any(nonzero_counts[c] > 0 for c in present_displacement)
    stress_nonzero = any(nonzero_counts[c] > 0 for c in present_stress)
    physics_nonzero = any(nonzero_counts[c] > 0 for c in present_physics)
    n_defect_nodes = sum(count for label, count in labels.items() if label not in {"0", "0.0", "healthy"})

    warnings: List[str] = []
    if missing_required:
        warnings.append("nodes.csv missing required columns: " + ",".join(missing_required))
    if not present_displacement:
        warnings.append("no displacement columns present")
    elif not displacement_nonzero:
        warnings.append("all displacement columns are zero")
    if not present_stress:
        warnings.append("no stress-equivalent columns present")
    elif not stress_nonzero:
        warnings.append("all stress-equivalent columns are zero")
    if present_physics and not physics_nonzero:
        warnings.append("all physics columns are zero")

    return {
        "columns": columns,
        "missing_required_columns": missing_required,
        "present_physics_columns": present_physics,
        "present_displacement_columns": present_displacement,
        "present_stress_columns": present_stress,
        "n_nodes": n_nodes,
        "label_counts": dict(labels),
        "n_defect_nodes": n_defect_nodes,
        "nonzero_counts": dict(nonzero_counts),
        "max_abs": maxima,
        "min": minima,
        "coordinate_bounds": coordinate_bounds,
        "warnings": warnings,
    }


def audit_dataset(root: Path) -> Dict[str, object]:
    sample_dirs = _discover_sample_dirs(root)
    samples = []
    warning_count = 0
    total_nodes = 0
    total_defect_nodes = 0
    schemas: Counter[Tuple[str, ...]] = Counter()

    for sample_dir in sample_dirs:
        node_report = _audit_nodes(sample_dir)
        n_elements, element_columns, element_warnings = _count_elements(sample_dir)
        metadata = _read_metadata(sample_dir)
        sample_warnings = list(node_report["warnings"]) + element_warnings

        metadata_defect_nodes = metadata.get("n_defect_nodes")
        if metadata_defect_nodes is not None:
            try:
                meta_count = int(float(metadata_defect_nodes))
                if meta_count != node_report["n_defect_nodes"]:
                    sample_warnings.append(
                        "metadata n_defect_nodes=%d but nodes.csv has %d"
                        % (meta_count, node_report["n_defect_nodes"])
                    )
            except ValueError:
                sample_warnings.append("metadata n_defect_nodes is not numeric")

        if metadata.get("defect_type", "healthy") != "healthy" and node_report["n_defect_nodes"] == 0:
            sample_warnings.append("defective metadata but zero defect-labeled nodes")

        total_nodes += int(node_report["n_nodes"])
        total_defect_nodes += int(node_report["n_defect_nodes"])
        warning_count += len(sample_warnings)
        schemas[tuple(node_report["columns"])] += 1

        samples.append(
            {
                "sample": str(sample_dir.relative_to(root)),
                "metadata": metadata,
                "nodes": node_report,
                "n_elements": n_elements,
                "element_columns": element_columns,
                "warnings": sample_warnings,
            }
        )

    return {
        "root": str(root),
        "n_samples": len(samples),
        "total_nodes": total_nodes,
        "total_defect_nodes": total_defect_nodes,
        "warning_count": warning_count,
        "schema_count": len(schemas),
        "schemas": [{"columns": list(k), "n_samples": v} for k, v in schemas.items()],
        "samples": samples,
    }


def _write_markdown(report: Dict[str, object], output: Path) -> None:
    lines = [
        "# Dataset Reliability Audit",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Root | `{report['root']}` |",
        f"| Samples | {report['n_samples']} |",
        f"| Total nodes | {report['total_nodes']} |",
        f"| Total defect nodes | {report['total_defect_nodes']} |",
        f"| Distinct node schemas | {report['schema_count']} |",
        f"| Warning count | {report['warning_count']} |",
        "",
        "## Samples",
        "",
        "| Sample | Nodes | Elements | Defect nodes | Labels | Warnings |",
        "|---|---:|---:|---:|---|---|",
    ]
    for sample in report["samples"]:
        nodes = sample["nodes"]
        warnings = "<br>".join(sample["warnings"]) if sample["warnings"] else "-"
        labels = ", ".join(f"{k}:{v}" for k, v in sorted(nodes["label_counts"].items()))
        lines.append(
            "| `{sample}` | {n_nodes} | {n_elements} | {n_defect} | {labels} | {warnings} |".format(
                sample=sample["sample"],
                n_nodes=nodes["n_nodes"],
                n_elements=sample["n_elements"],
                n_defect=nodes["n_defect_nodes"],
                labels=labels,
                warnings=warnings,
            )
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit FEM CSV dataset reliability")
    parser.add_argument("roots", nargs="+", type=Path, help="Dataset roots to audit")
    parser.add_argument("--json", type=Path, default=None, help="Write JSON report")
    parser.add_argument("--markdown", type=Path, default=None, help="Write Markdown report for the first root")
    parser.add_argument("--fail-on-warning", action="store_true", help="Exit non-zero if warnings are found")
    args = parser.parse_args()

    reports = [audit_dataset(root) for root in args.roots]
    if args.json:
        payload = reports[0] if len(reports) == 1 else {"reports": reports}
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.markdown:
        _write_markdown(reports[0], args.markdown)

    for report in reports:
        print(
            "{root}: samples={n_samples} nodes={total_nodes} defect_nodes={total_defect_nodes} "
            "schemas={schema_count} warnings={warning_count}".format(**report)
        )
        for sample in report["samples"]:
            if sample["warnings"]:
                print("  - %s" % sample["sample"])
                for warning in sample["warnings"]:
                    print("    * " + warning)

    if args.fail_on_warning and any(report["warning_count"] for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
