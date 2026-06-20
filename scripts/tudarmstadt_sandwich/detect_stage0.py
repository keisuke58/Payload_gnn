#!/usr/bin/env python3
"""Channel-aware locally-detrended Stage-0 detection on the TU Darmstadt
real sandwich-panel 200 kHz air-coupled-UT data (doi:10.48328/tudatalib-2000).

This is the fairing's real **sandwich** proxy for the structure-agnostic SHM
paper (§5.4). Result: AUROC 0.90 (global null) / 0.85 (strict local-healthy
null), CH6 0.88, CH7 0.97, n_def=28. See NOTES.md for the full story.

Code is version-controlled here; the raw dataset (~144 MB core, gitignored)
lives under data/external/tudarmstadt_sandwich/. Override with $TUD_SANDWICH_DATA.

Run:  LD_LIBRARY_PATH=/home/nishioka/miniconda3/lib \
      /home/nishioka/miniconda3/bin/python3.12 detect_stage0.py
Needs clean_windows.json (committed alongside this script; regenerate with
parse_windows.py) and the .mat data in $TUD_SANDWICH_DATA.
"""
import os, json, numpy as np, scipy.io as sio
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("TUD_SANDWICH_DATA",
                      os.path.abspath(os.path.join(HERE, "..", "..", "data", "external", "tudarmstadt_sandwich")))
HW = 500                      # +/- search tolerance (idx) for the documented +-5-15cm marker offset
DETREND_W = 2001              # per-channel running-mean window to remove slow air-coupled drift
rng = np.random.default_rng(1)


def load():
    m = sio.loadmat(os.path.join(DATA, "eval_defect_200khz_inline_data_all_CH.mat"), squeeze_me=True)
    A = m['meas_amplitudes'].astype(float)
    T = m['tt_corr_times'].astype(float)
    Ch = list(m['Channels'].astype(int))
    for X in (A, T):                                   # fill the 0.8% NaNs per channel
        for k in range(X.shape[1]):
            col = X[:, k]; col[np.isnan(col)] = np.nanmedian(col)
    return A, T, Ch


def zmag_of(A, T):
    det = lambda X: X - uniform_filter1d(X, size=DETREND_W, axis=0, mode='nearest')
    rz = lambda R: np.abs((R - np.median(R, 0)) / (1.4826 * (np.median(np.abs(R - np.median(R, 0)), 0) + 1e-12)))
    return np.maximum(rz(det(A)), rz(det(T)))          # per-position per-channel local deviation


def auroc_vs_pool(scores, chs, NULL):
    return float(np.mean([(s > NULL[ch]).mean() for s, ch in zip(scores, chs)]))


def main():
    A, T, Ch = load()
    chcol = {c: k for k, c in enumerate(Ch)}
    z = zmag_of(A, T); N = z.shape[0]
    W = json.load(open(os.path.join(HERE, "clean_windows.json")))
    centers = sorted((w['s'] + w['e']) // 2 for w in W)
    wmax = lambda zc, c: float(np.max(zc[max(0, c - HW):min(N, c + HW)]))

    # defect scores in each defect's own channel
    ds, dch = [], []
    for w in W:
        ch, c = w['ch'], (w['s'] + w['e']) // 2
        if ch > 0:
            ds.append(wmax(z[:, chcol[ch]], c)); dch.append(ch)
        elif ch == -1:                                  # "all CHs" -> best channel
            k = max(range(len(Ch)), key=lambda k: wmax(z[:, k], c))
            ds.append(wmax(z[:, k], c)); dch.append(Ch[k])
    ds = np.array(ds)

    def null_pool(region):
        NULL = {}
        for ch in set(dch):
            xs = []
            while len(xs) < 400:
                c = int(rng.integers(*region))
                if min(abs(c - cc) for cc in centers) > HW + 600:
                    xs.append(wmax(z[:, chcol[ch]], c))
            NULL[ch] = np.array(xs)
        return NULL

    lo, hi = min(centers) - 3000, max(centers) + 3000
    glob = null_pool((HW + 1000, N - HW - 1000))
    loc = null_pool((lo, hi))
    a_glob, a_loc = auroc_vs_pool(ds, dch, glob), auroc_vs_pool(ds, dch, loc)
    bs = [auroc_vs_pool(ds[i], [dch[j] for j in i], glob)
          for i in (rng.integers(0, len(ds), len(ds)) for _ in range(2000))]
    ci = [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]
    thr = {ch: np.quantile(glob[ch], 0.95) for ch in glob}
    det5 = float(np.mean([s > thr[ch] for s, ch in zip(ds, dch)]))
    res = {"HW": HW, "n_def": len(ds), "AUROC_global": a_glob, "CI90": ci,
           "AUROC_local_strict": a_loc, "det_at_5FA_global": det5,
           "per_channel": {str(t): auroc_vs_pool(ds[[i for i, c in enumerate(dch) if c == t]],
                                                  [t] * dch.count(t), glob) for t in (6, 7)}}
    json.dump(res, open(os.path.join(DATA, "stage0_result.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    c6 = z[:, chcol[6]]
    ax1.plot(c6, lw=0.4, color='#333', alpha=0.8)
    for w in W:
        if w['ch'] in (6, -1):
            ax1.axvspan(w['s'] - HW, w['e'] + HW, color='#2a9d4a', alpha=0.18)
    ax1.axhline(thr[6], color='#d7301f', ls='--', lw=1, label='5% FA threshold')
    ax1.set(title='CH6 local deviation (z) — green = labelled defect ±search',
            xlabel='along-track position index', ylabel='|z| (amp or ToF)', ylim=(0, 12))
    ax1.legend(fontsize=8)
    ths = np.linspace(0, max(ds.max(), max(v.max() for v in glob.values())), 200)
    tpr = [(ds > t).mean() for t in ths]
    fpr = [float(np.mean([(glob[ch] > t).mean() for ch in dch])) for t in ths]
    ax2.plot(fpr, tpr, color='#2c7fb8', lw=2, label=f'AUROC={a_glob:.3f}')
    ax2.plot([0, 1], [0, 1], 'k:', lw=0.8)
    ax2.set(title='Stage-0 detection ROC (real sandwich, 200 kHz)',
            xlabel='false-alarm rate', ylabel='detection rate'); ax2.legend(fontsize=9)
    fig.suptitle('TU Darmstadt sandwich panel — channel-aware locally-detrended Stage-0 detection')
    fig.tight_layout(); fig.savefig(os.path.join(DATA, 'stage0_detection.png'), dpi=160, bbox_inches='tight')
    print("wrote stage0_detection.png, stage0_result.json to", DATA)


if __name__ == "__main__":
    main()
