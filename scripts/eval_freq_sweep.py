import os,sys,glob,numpy as np,torch
# CAVEAT: debond zips use different Vpp for 50/200kHz (healthy 10/4.5Vpp vs defect 11/4Vpp).
# Only 100kHz(9Vpp) and 300kHz(3Vpp) are Vpp-matched -> fair. Also: effectively n=3 specimens.
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import eval_panel_gw as E
from sklearn.metrics import roc_auc_score, average_precision_score
from build_gw_graph import extract_time_features, build_edge_index
from parse_ogw_wavefield import load_ogw
from torch_geometric.data import Data
OGWX=os.path.join(HERE,"..","data","external","ogw_stringer","extracted")
SCEN=[("Intact",0),("FirstImpact",1),("SecondImpact",1)]
FREQ={"50kHz":"BURST_50kHz*","100kHz":"BURST_100kHz*","200kHz":"BURST_200kHz*","300kHz":"BURST_300kHz*"}
rng=np.random.default_rng(0); NB=40; NS=9
m=E.load_model()
def boot(pat):
    gs=[]
    for tag,y in SCEN:
        f=sorted(glob.glob(os.path.join(OGWX,f"OGW_CFRP_Stringer_Wavefield_{tag}",pat)))
        if not f: continue
        coords,wf,t=load_ogw(f[0]); N=coords.shape[0]
        for _ in range(NB):
            si=rng.choice(N,NS,replace=False); sd={i:wf[:,s] for i,s in enumerate(si)}
            ft=np.asarray(extract_time_features(t,sd,"comprehensive"),dtype=np.float32)
            ei=build_edge_index(NS,[float(coords[s,0]*1000) for s in si],"full")
            gs.append(Data(x=torch.tensor(ft),edge_index=ei,y=torch.tensor([y])))
        del wf
    return gs
print(f"{'freq':8s}{'N':>5s}{'AUC':>7s}{'AUPRC':>7s}{'rec@FPR5%':>10s}{'bal-acc':>8s}")
for fk,pat in FREQ.items():
    g=boot(pat)
    if not g: print(f"{fk:8s} (no data)"); continue
    yt,yp,pd=E.run(m,g)
    auc=roc_auc_score(yt,pd) if yt.any() and (yt==0).any() else float('nan')
    aup=average_precision_score(yt,pd)
    hs=pd[yt==0]; ds=pd[yt==1]; thr=np.percentile(hs,95); rec=(ds>=thr).mean(); fpr=(hs>=thr).mean()
    print(f"{fk:8s}{len(g):5d}{auc:7.3f}{aup:7.3f}{rec:10.3f}{0.5*(rec+1-fpr):8.3f}",flush=True)
