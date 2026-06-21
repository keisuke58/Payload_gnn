# abaqus python scripts/check_odb_maxstress.py
# Report max von Mises (last frame) for candidate static H3 odbs.
# Lets us delete only the genuinely zero-stress (failed cohesive) results.
from odbAccess import openOdb
import glob, os

try:
    import numpy as np
    HAVE_NP = True
except Exception:
    HAVE_NP = False

dirs = ['abaqus_work', 'abaqus_work_f02', 'abaqus_work_f03', '.']
seen = set()
paths = []
for d in dirs:
    for p in glob.glob(os.path.join(d, 'H3_*.odb')):
        rp = os.path.normpath(p)
        if rp in seen:
            continue
        seen.add(rp)
        base = os.path.basename(rp)
        if base.startswith('Job-GW') or base.startswith('H3_healthy_a2'):
            continue
        paths.append(rp)

paths.sort()
print('CHECKING %d odbs' % len(paths))
for p in paths:
    try:
        odb = openOdb(p, readOnly=True)
        if len(odb.steps) == 0:
            print('%s\tNO_STEPS' % p); odb.close(); continue
        step = list(odb.steps.values())[-1]
        if len(step.frames) == 0:
            print('%s\tNO_FRAMES' % p); odb.close(); continue
        fr = step.frames[-1]
        if 'S' not in fr.fieldOutputs.keys():
            print('%s\tNO_S\tframes=%d' % (p, len(step.frames))); odb.close(); continue
        s = fr.fieldOutputs['S']
        mx = 0.0
        for b in s.bulkDataBlocks:
            m = getattr(b, 'mises', None)
            if m is None:
                continue
            if HAVE_NP:
                if len(m):
                    v = float(np.max(m))
                    if v > mx:
                        mx = v
            else:
                for x in m:
                    if x > mx:
                        mx = x
        print('%s\tMAXMISES=%.4g\tnframes=%d' % (p, mx, len(step.frames)))
        odb.close()
    except Exception as e:
        print('%s\tERROR\t%s' % (p, str(e)[:60]))
print('DONE')
