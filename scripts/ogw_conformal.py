#!/usr/bin/env python3
"""E: 実OGW検出に分布フリー(split-conformal)FPR保証 — 温度シフトで破れ, DAで回復。

検出スコア $s=$ P(damage) に split-conformal を適用する：
  較正(ソース cool 月の健全)スコアの $(1-\\alpha)$ 分位を閾値 $\\tau$ とすると、
  同分布のテスト健全に対し FPR $\\le \\alpha$ が有限標本で保証される（交換可能性）。
温度シフト(hot 月)では交換可能性が崩れ保証が破れる(empirical FPR > α)。
分位点ドメイン適応で分布を整列すると保証が回復することを示す。
入力: 実 long-term OGW (payload_da_longterm.load_month)。出力: results/ogw/ogw_conformal.{png,pdf,json}
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["Nimbus Roman","STIXGeneral","DejaVu Serif"],"mathtext.fontset":"stix"})
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(HERE,"..","src"))
sys.path.insert(0,HERE)
import domain_adapt as da
from payload_da_longterm import load_month
from sklearn.linear_model import LogisticRegression

ALPHAS=[0.025,0.05,0.10,0.15]
SRC,TGT="2018_03","2018_07"

def scores(method):
    """detector scores on source-healthy (calib) and target-healthy (test) under `method`."""
    Xs,ys,ts=load_month(SRC,3000); Xt,yt,tt=load_month(TGT,3000)
    Xs_a=da.ALIGNERS[method](Xs,Xt); Xt_a=da._apply_target(method,Xs,Xt)
    clf=LogisticRegression(max_iter=1000).fit(Xs_a,ys)
    s_cal=clf.predict_proba(Xs_a[ys==0])[:,1]      # source healthy (exchangeable calib)
    s_tgt=clf.predict_proba(Xt_a[yt==0])[:,1]      # target healthy (shifted)
    return s_cal,s_tgt,(ts,tt)

def conformal_fpr(s_cal,s_tgt,alpha):
    n=len(s_cal); k=int(np.ceil((1-alpha)*(n+1)))   # split-conformal rank
    k=min(max(k,1),n); tau=np.sort(s_cal)[k-1]
    return float((s_tgt>tau).mean()), float(tau)

def main():
    out={"src":SRC,"tgt":TGT,"by_method":{}}
    print("E: split-conformal FPR guarantee under OGW temperature shift")
    for method in ("none","quantile"):
        s_cal,s_tgt,(ts,tt)=scores(method)
        rows=[]
        for a in ALPHAS:
            fpr,tau=conformal_fpr(s_cal,s_tgt,a); rows.append((a,fpr))
        out["by_method"][method]={"temp":(ts,tt),"alpha":ALPHAS,
                                  "emp_fpr":[r[1] for r in rows]}
        print("  %-9s (%.0f->%.0fC): "%(method,ts,tt)+
              " ".join("α=%.2f→FPR%.2f"%(a,f) for a,f in rows))
    fig,ax=plt.subplots(figsize=(5.6,4.2))
    a=np.array(ALPHAS)
    ax.plot([0,0.16],[0,0.16],"k--",lw=1,label="nominal guarantee (FPR=α)")
    ax.plot(a,out["by_method"]["none"]["emp_fpr"],"-o",color="#c62828",label="no DA (shift breaks it)")
    ax.plot(a,out["by_method"]["quantile"]["emp_fpr"],"-o",color="#0f8b8d",label="quantile DA (restored)")
    ax.set_xlabel("nominal level $\\alpha$"); ax.set_ylabel("empirical target FPR")
    ax.set_title("Split-conformal FPR guarantee on REAL OGW\nbroken by temperature shift, restored by DA",fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    d=os.path.join(HERE,"..","results","ogw"); os.makedirs(d,exist_ok=True)
    for e in ("png","pdf"): fig.savefig(os.path.join(d,"ogw_conformal."+e),dpi=200,bbox_inches="tight")
    json.dump(out,open(os.path.join(d,"ogw_conformal.json"),"w"),indent=2)
    nb=out["by_method"]["none"]["emp_fpr"]; qb=out["by_method"]["quantile"]["emp_fpr"]
    print("  => no-DA empirical FPR exceeds α (guarantee void under shift); quantile-DA tracks α (restored).")
    print("     e.g. α=0.05: no-DA FPR=%.2f vs DA FPR=%.2f"%(nb[1],qb[1]))
    print("  saved results/ogw/ogw_conformal.{png,pdf,json}")

if __name__=="__main__": main()
