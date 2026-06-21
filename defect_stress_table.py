# defect_stress_table.py - quantitative defect stress-concentration table for H3 scenarios.
# Reads loaded (Max-Q) odbs, writes a markdown 考察 table (peak Mises/DSPSS, SCF vs healthy).
# Run: abq2024 python defect_stress_table.py
from odbAccess import openOdb
from abaqusConstants import *
import numpy as np, os
WD='/home/nishioka/Payload2026'
# (label, odb path) - healthy first for SCF baseline
SCN=[('Healthy','H3_healthy_mq'),('Debond(outer)','H3_debond_mq'),('InnerDebond','abaqus_work/H3_InnerDebond_t2'),
     ('Delamination','H3_delam_mq'),('FOD','H3_fod_mq'),('Impact','H3_impact_mq'),
     ('Thermal','H3_Thermal_t2'),('Acoustic','H3_acoustic_mq')]
def peak(path):
    o=openOdb(path+'.odb',readOnly=True)
    gm=0.0; dmax=-1e30; dmin=1e30; inst=''
    for st in o.steps.values():
        for fr in st.frames:
            if 'S' not in fr.fieldOutputs: continue
            S=fr.fieldOutputs['S']
            for blk in S.getScalarField(invariant=MISES).bulkDataBlocks:
                if blk.data is not None and len(blk.data):
                    mx=float(np.max(blk.data))
                    if mx>gm: gm=mx; inst=(blk.instance.name if blk.instance else '?')
            for blk in S.getScalarField(invariant=PRESS).bulkDataBlocks:
                if blk.data is not None and len(blk.data):
                    ds=-3.0*np.asarray(blk.data); dmax=max(dmax,float(ds.max())); dmin=min(dmin,float(ds.min()))
    o.close(); return gm,dmax,dmin,inst
rows=[]; base=None
for label,p in SCN:
    fp=os.path.join(WD,p+'.odb')
    if not os.path.exists(fp): rows.append((label,None)); continue
    try:
        r=peak(os.path.join(WD,p)); rows.append((label,r))
        if label=='Healthy': base=r[0]
    except Exception as e: rows.append((label,('ERR',str(e)[:40],0,'')))
out=['# H3 fairing 欠陥シナリオ 応力集中 考察 (Max-Q load)','',
     '同一構造・同一欠陥位置(z=3000,θ=45,r=200)・同一荷重(Max-Q)で defect_type のみ変化。',
     'SCF = ピークMises / 健全ピークMises。','',
     '| シナリオ | peak Mises [MPa] | DSPSS+ [MPa] | DSPSS- [MPa] | SCF | peak部位 |',
     '|---|---|---|---|---|---|']
for label,r in rows:
    if r is None: out.append('| %s | (odb未生成) | | | | |'%label); continue
    if r[0]=='ERR': out.append('| %s | ERR %s | | | | |'%(label,r[1])); continue
    scf=(r[0]/base) if base else 0
    out.append('| %s | %.1f | %.1f | %.1f | %.2f | %s |'%(label,r[0],r[1],r[2],scf,r[3]))
open(os.path.join(WD,'H3_defect_stress_kousatsu.md'),'w').write('\n'.join(out)+'\n')
print('\n'.join(out))
