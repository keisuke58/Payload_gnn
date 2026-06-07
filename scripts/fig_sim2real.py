import os,sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/nishioka/git/wccm2026-cfrp-gnn/slides/figure_sources")
try:
    import thesis_style as ts; ts.use(); matplotlib.rcParams.update({"font.size":11,"axes.titlesize":12})
except Exception:
    plt.rcParams.update({"font.family":"serif","font.size":11})
from sklearn.metrics import roc_curve, auc as sk_auc
import eval_panel_gw as E
m=E.load_model(); g=E.ogw_boot()
yt,yp,pd=E.run(m,g)
NAVY="#142b54"; ORN="#c0712c"
fig,ax=plt.subplots(1,2,figsize=(9.2,3.6))
# (a) score distribution
hs=pd[yt==0]; ds=pd[yt==1]
bins=np.linspace(0,1,26)
ax[0].hist(hs,bins=bins,alpha=0.7,color=NAVY,label="healthy (intact)")
ax[0].hist(ds,bins=bins,alpha=0.7,color=ORN,label="defect (debond)")
thr=np.percentile(hs,95)
ax[0].axvline(thr,ls="--",lw=1.2,color="k",label=f"calibrated thr (FPR5\\%)={thr:.2f}")
ax[0].set_xlabel("predicted P(defect)"); ax[0].set_ylabel("count")
ax[0].set_title("OGW score separation (AUC 0.83)")
ax[0].legend(frameon=False,fontsize=8); ax[0].spines["top"].set_visible(False); ax[0].spines["right"].set_visible(False)
# (b) ROC + operating point
fpr,tpr,_=roc_curve(yt,pd); a=sk_auc(fpr,tpr)
ax[1].plot(fpr,tpr,color=NAVY,lw=1.8,label=f"ROC (AUC={a:.2f})")
ax[1].plot([0,1],[0,1],ls=":",color="0.5",lw=0.8)
# operating point at FPR~5%
rec5=float((ds>=thr).mean()); fpr5=float((hs>=thr).mean())
ax[1].scatter([fpr5],[rec5],s=70,color=ORN,zorder=3,edgecolors="k",linewidths=0.6,
              label=f"op. point: recall={rec5:.2f} @ FPR={fpr5:.2f}")
ax[1].set_xlabel("false positive rate"); ax[1].set_ylabel("recall (1$-$FNR)")
ax[1].set_title("Detection transfers; threshold calibrated"); ax[1].set_xlim(0,1); ax[1].set_ylim(0,1.02)
ax[1].legend(frameon=False,fontsize=8,loc="lower right"); ax[1].spines["top"].set_visible(False); ax[1].spines["right"].set_visible(False)
fig.suptitle("Payload sim-to-real: FEM-trained GNN $\\to$ real OGW debond  "
             "(naive acc 0.70 $\\to$ AUC 0.83 $\\to$ calibrated bal-acc 0.83)",fontsize=11)
fig.tight_layout()
os.makedirs(f"{HERE}/../results/ogw",exist_ok=True)
fig.savefig(f"{HERE}/../results/ogw/fig_sim2real_3stage.png",dpi=300)
print("saved fig_sim2real_3stage.png  (AUC %.3f, op recall %.3f @ FPR %.3f)"%(a,rec5,fpr5))
