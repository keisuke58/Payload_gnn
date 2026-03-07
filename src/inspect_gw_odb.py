# -*- coding: utf-8 -*-
"""Quick inspect of GW ODB history regions."""
import sys
from odbAccess import openOdb

odb_path = sys.argv[1]
odb = openOdb(path=odb_path, readOnly=True)
step = odb.steps.values()[0]

print("Step: %s" % step.name)
print("History regions: %d" % len(step.historyRegions.keys()))
for i, key in enumerate(step.historyRegions.keys()):
    region = step.historyRegions[key]
    outputs = region.historyOutputs.keys()
    n_data = 0
    if len(outputs) > 0:
        first_out = region.historyOutputs[outputs[0]]
        n_data = len(first_out.data)
    print("  [%d] %s -> outputs=%s, n_data=%d" % (i, key, list(outputs), n_data))
    if i > 20:
        print("  ... (truncated)")
        break

odb.close()
