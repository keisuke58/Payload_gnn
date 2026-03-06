#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_gw_odb.py — 単一 ODB の適切性チェック（Abaqus Python 用）

1解析終了後に ODB が正常かチェック。
- ファイル存在・サイズ
- ODB オープン可能
- Step 完了・History 出力あり

Usage (Abaqus Python):
  abaqus python scripts/verify_gw_odb.py Job-GW-Fair-Healthy.odb
  abaqus python scripts/verify_gw_odb.py Job-GW-Fair-0000.odb
"""
import sys
import os

# ODB は Abaqus Python から開く
try:
    from odbAccess import openOdb
except ImportError:
    print("ERROR: Run with 'abaqus python' (odbAccess required)")
    sys.exit(1)

MIN_SIZE = 500000   # 500KB
MIN_SENSORS = 5
MIN_FRAMES = 50


def verify_odb(odb_path):
    """Check ODB. Returns (ok: bool, msg: str)."""
    if not os.path.exists(odb_path):
        return False, "ODB missing"
    size = os.path.getsize(odb_path)
    if size < MIN_SIZE:
        return False, "ODB too small (%d < %d)" % (size, MIN_SIZE)

    try:
        odb = openOdb(odb_path, readOnly=True)
    except Exception as e:
        return False, "ODB open failed: %s" % str(e)[:80]

    try:
        steps = odb.steps
        if not steps:
            return False, "No steps"
        last_step = list(steps.keys())[-1]
        step = steps[last_step]
        frames = step.frames
        if len(frames) < MIN_FRAMES:
            return False, "Too few frames (%d < %d)" % (len(frames), MIN_FRAMES)

        # History regions (sensor outputs)
        hr = step.historyRegions
        n_hist = len(hr)
        if n_hist < MIN_SENSORS:
            return False, "Too few history regions (%d < %d)" % (n_hist, MIN_SENSORS)

        odb.close()
        return True, "OK (sensors=%d, frames=%d)" % (n_hist, len(frames))
    except Exception as e:
        try:
            odb.close()
        except Exception:
            pass
        return False, "Check failed: %s" % str(e)[:80]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: abaqus python verify_gw_odb.py <odb_path>")
        sys.exit(1)
    odb_path = sys.argv[1]
    ok, msg = verify_odb(odb_path)
    print(msg)
    sys.exit(0 if ok else 1)
