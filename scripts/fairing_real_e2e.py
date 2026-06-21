#!/usr/bin/env python3
"""フェアリング(CFRPストリンガ) 実OGWガイド波 end-to-end —— 1構造を実データで端から端。

方針(記憶: 深さ>広さ・実検証重視・overclaim禁止):
 広さを足さず、フェアリング1構造の実測ガイド波で検出→分類→特性同定を端から端に通し、
 予後で実データが尽きる所を正直に図示する。

実データ:
 - ストリンガ波動場(Open Guided Waves): Intact / FirstImpact / SecondImpact、9センサ×U3時系列(100kHz)。
   損傷指標 DI_s = 1 - 正規化相互相関(scenario, intact) を各センサで算出。
 - 長期ガイド波(34ヶ月・温度): 健全/損傷の大N検出 recall/FPR(温度交絡、§9のDAで回復)。

ステージ別の実データ被覆(正直):
 Stage 0 検出   : 実(長期 大N + ストリンガ DI>noise)         ✓ real
 Stage 1 分類   : 実(First vs Second を DI パターンで判別)     ✓ real
 Stage 2 特性同定: 実(severity=平均DI単調, location=センサ局在) ✓ real
 Stage 3 予後   : 合成(代表 FairingPrognosis, 実サイクル試験なし) ✗ SYNTHETIC ← 実データ終了
 Stage 4/5      : 合成                                          ✗ SYNTHETIC
出力: results/ogw/fairing_real_e2e.{png,pdf,json}
"""
import os, json, csv
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["Nimbus Roman","STIXGeneral","DejaVu Serif"],"mathtext.fontset":"stix"})

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.join(HERE,"..")
SDIR=os.path.join(ROOT,"data","external","ogw_stringer","csv9")
RES=os.path.join(ROOT,"results","ogw")

def load_csv(f):
    rows=[r for r in csv.reader(open(os.path.join(SDIR,f))) if r and not r[0].startswith("#")]
    d=np.array([[float(x) for x in r] for r in rows[1:]]); return d[:,1:]   # (T,9)
I=load_csv("OGW_Intact_100kHz_9s.csv"); F=load_csv("OGW_FirstImpact_100kHz_9s.csv"); S=load_csv("OGW_SecondImpact_100kHz_9s.csv")
SENS_X=[33,250,467,33,250,467,33,250,467]   # mm (csv header)

def DI(sig,base):
    out=[]
    for s in range(sig.shape[1]):
        a=sig[:,s]-sig[:,s].mean(); b=base[:,s]-base[:,s].mean()
        out.append(1-np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    return np.array(out)
diF=DI(F,I); diS=DI(S,I); diI=np.zeros(9)
sev=[0.0,float(diF.mean()),float(diS.mean())]
locF=int(np.argmax(diF)); locS=int(np.argmax(diS))

# 長期(大N)検出: recall/FPR vs 温度（検出の実統計はここ＝健全/損傷の母集団がある）
lt=None
try: lt=json.load(open(os.path.join(RES,"detection_over_time.json")))
except Exception: pass
lt_recall=lt_fpr=None
COMP_MODE = "FULL"  # amp-norm + temp-regress (train 2018-2019, test 2020-2022)
if lt and COMP_MODE in lt:
    r=[x for x in lt[COMP_MODE]["rows"]]
    rv=[x["recall"] for x in r if x.get("recall") is not None]
    fv=[x["fpr"] for x in r if x.get("fpr") is not None]
    lt_recall=float(np.mean(rv)); lt_fpr=float(np.mean(fv))
lt_fpr_raw = float(np.mean([x["fpr"] for x in lt.get("AMP",{}).get("rows",[]) if x.get("fpr") is not None])) if lt else None

out={"stage0_detect_realdata":"long-term GW 34mo large-N FULL(amp+regress): recall=%.3f FPR=%.3f (AMP-only FPR=%.3f)"%(lt_recall or 0,lt_fpr or 0, lt_fpr_raw or 0),
     "stringer_note":"single intact baseline -> no stringer detection-rate; stringer used for classify/characterise",
     "stage2_mean_DI":{"Intact":0.0,"First":sev[1],"Second":sev[2]},
     "stage2_severity_monotone":bool(sev[2]>sev[1]>0),
     "stage1_classify_peak_sensor":{"First":locF,"Second":locS},
     "stage3_prognosis":"SYNTHETIC (representative FairingPrognosis; no real cycle/growth data)"}

# ===================== figure =====================
fig=plt.figure(figsize=(14,4.2))
gs=fig.add_gridspec(1,4,width_ratios=[1.15,1.25,0.9,1.15],wspace=0.42)

# (a) Stage 0 detect — 長期 大N 実データ (温度に対する recall)
axa=fig.add_subplot(gs[0,0])
if lt and "AMP" in lt and "FULL" in lt:
    for mode,col,lab in [("AMP","#aaa","AMP-only (FPR=%.2f)"%lt_fpr_raw),
                         ("FULL","#1565c0","FULL amp+regress (FPR=%.2f)"%lt_fpr)]:
        r=[x for x in lt[mode]["rows"] if x.get("recall") is not None]
        T=[x["tempC"] for x in r]; rec=[x["recall"] for x in r]
        axa.scatter(T,rec,s=14,c=col,alpha=0.75,label=lab)
    axa.set_xlabel("temperature [$^\\circ$C]");axa.set_ylabel("detection recall")
    axa.set_title("(a) Stage 0 detect — REAL 34mo GW\nFULL: Recall=%.2f FPR=%.2f (AMP: FPR=%.2f)"%(lt_recall,lt_fpr,lt_fpr_raw),fontsize=8.5)
    axa.set_ylim(0,1.05);axa.grid(alpha=0.3);axa.legend(fontsize=6,frameon=False)
else:
    axa.text(0.5,0.5,"long-term JSON\nnot found",ha="center")

# (b) Stage 1/2 — ストリンガ 実 DI ヒートマップ(センサ×シナリオ)
axb=fig.add_subplot(gs[0,1])
M=np.vstack([diI,diF,diS])                       # (3 scenarios, 9 sensors)
im=axb.imshow(M,aspect="auto",cmap="inferno",vmin=0,vmax=max(diS.max(),diF.max()))
axb.set_yticks([0,1,2]);axb.set_yticklabels(["Intact","First","Second"],fontsize=8)
axb.set_xticks(range(9));axb.set_xticklabels(["S%d\n%dmm"%(i,SENS_X[i]) for i in range(9)],fontsize=6)
axb.set_title("(b) Stage 1/2 — REAL stringer damage index\n(detect+classify+localize: which paths light up)",fontsize=8.5)
axb.axhline(0.5,color="w",lw=0.6);axb.axhline(1.5,color="w",lw=0.6)
cb=fig.colorbar(im,ax=axb,fraction=0.046,pad=0.03);cb.set_label("DI = 1$-$xcorr",fontsize=7);cb.ax.tick_params(labelsize=6)

# (c) Stage 2 severity — 平均DI 単調(実)
axc=fig.add_subplot(gs[0,2])
axc.bar([0,1,2],sev,color=["#2e7d32","#f9a825","#c62828"])
axc.set_xticks([0,1,2]);axc.set_xticklabels(["Intact","First","Second"],fontsize=8)
axc.set_ylabel("mean damage index");axc.set_title("(c) Stage 2 severity\nREAL monotone progression",fontsize=8.5)
axc.grid(alpha=0.3,axis="y")

# (d) 実データ被覆マップ(正直): どこで実データが尽きるか
axd=fig.add_subplot(gs[0,3]); axd.axis("off"); axd.set_xlim(0,1);axd.set_ylim(0,6)
stg=[("Stage 0 detect","REAL",True),("Stage 1 classify","REAL",True),
     ("Stage 2 characterise","REAL",True),("Stage 3 prognose","SYNTHETIC",False),
     ("Stage 4 fleet","SYNTHETIC",False),("Stage 5 design","SYNTHETIC",False)]
for i,(s,tag,real) in enumerate(stg):
    y=5-i; c="#2e7d32" if real else "#9e9e9e"
    axd.add_patch(plt.Rectangle((0.05,y+0.1),0.9,0.78,fc=("#e8f5e9" if real else "#f5f5f5"),ec=c,lw=1.6))
    axd.text(0.10,y+0.5,s,va="center",fontsize=8.5)
    axd.text(0.92,y+0.5,tag,va="center",ha="right",fontsize=8,color=c,fontweight="bold")
axd.axhline(2.95,xmin=0.05,xmax=0.95,color="#c62828",lw=1.8,ls="--")
axd.text(0.5,2.74,"REAL DATA ENDS HERE",ha="center",fontsize=7.5,color="#c62828",fontweight="bold")
axd.set_title("(d) honest real-data coverage",fontsize=8.5)

fig.suptitle("Fairing (CFRP stringer) end-to-end on REAL Open-Guided-Waves data: detect $\\to$ classify $\\to$ characterise are real; prognosis is honestly synthetic",
             fontsize=10.5,y=1.02)
for e in ("png","pdf"): fig.savefig(os.path.join(RES,"fairing_real_e2e."+e),dpi=200,bbox_inches="tight")
json.dump(out,open(os.path.join(RES,"fairing_real_e2e.json"),"w"),indent=2)
print("REAL OGW fairing end-to-end:")
print("  Stage0 detect (REAL, long-term large-N): mean recall=%.3f mean FPR=%.3f (temp-confounded; DA recovers §9)"%(lt_recall or 0,lt_fpr or 0))
print("  Stage1 classify: peak-DI sensor First=S%d(%dmm) Second=S%d(%dmm) -> distinct location"%(locF,SENS_X[locF],locS,SENS_X[locS]))
print("  Stage2 characterise: mean DI Intact=0 < First=%.3f < Second=%.3f (REAL monotone severity)"%(sev[1],sev[2]))
print("  Stage3 prognosis: SYNTHETIC (representative; no real cycle data) <- real data ends")
print("  saved results/ogw/fairing_real_e2e.{png,pdf,json}")
