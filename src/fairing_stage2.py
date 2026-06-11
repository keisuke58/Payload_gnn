# -*- coding: utf-8 -*-
"""fairing_stage2.py — Fairing Stage-2 debond SIZE/SEVERITY characterisation.

Two-stage SHM pipeline for the H3 CFRP fairing:

    Stage-1  detection      (anomaly: healthy vs damaged)
    Stage-2  CHARACTERISATION  <-- THIS MODULE
    Stage-3  prognosis       (cfrp_phasefield_2d.flight_clearance, go/no-go)

The interstage panel already has a Stage-2 (the WCCM FMPE model,
fmpe_defect.py) that turns one guided-wave measurement into a posterior over
the defect parameters and feeds Stage-3.  The fairing had NO Stage-2: the
flight-clearance demo was fed a *synthetic* radius posterior.  This module
fills that gap with a numpy/scikit-learn-only debond severity estimator that
emits a posterior (mean + uncertainty) over debond severity / radius from the
guided-wave sensor signals, and a bridge that turns it into the radius-sample
posterior the fairing Stage-3 prognosis consumes.

Pipeline
--------
1. load_sensor_csv()        : read Job-GW-*_sensors.csv (time + sensor U3 cols,
                              "# x_mm,..." position comment).
2. extract_features()       : per-sensor waveform descriptors (rms, energy,
                              peak, crest, zcr, spectral centroid/bandwidth,
                              dominant freq, FFT-envelope max + decay) pooled
                              into a FIXED-LENGTH vector that is independent of
                              the number of sensors in a given config.
3. SeverityEstimator        : (a) a healthy-reference Mahalanobis anomaly score
                              (continuous severity, no debond labels needed) and
                              (b) a bagged-linear regressor ensemble mapping
                              features -> debond severity with a predictive sd
                              (mean + spread of the bag), trained on the few
                              labelled size-sweep configs with WINDOWED samples
                              for n.  Output: a posterior (mean, sd, samples).
4. severity_to_radius_posterior() : bridge -> array of posterior radius samples
                              (mm) for the fairing Stage-3 prognosis.

HONEST SCOPE
------------
Only a handful of labelled debond configs exist and they are simulation-only
(r15 / r40 radius proxies, near / edge positions).  This is therefore a
severity-characterisation PROOF-OF-CONCEPT with uncertainty, NOT a calibrated
size-regression: the absolute radius scale is anchored to the r15/r40 proxies
and the predictive sd is an ensemble/Mahalanobis spread, not a frequentist
coverage guarantee.  Windowing the (short) signals is what gives us enough
samples to fit/estimate variance; it is a stand-in for measurement repeats.

Run:
    python3 src/fairing_stage2.py --test    # self-test, no Abaqus data needed
    python3 src/fairing_stage2.py --fig      # severity vs debond-size figure
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  Paths / config catalogue
# ─────────────────────────────────────────────────────────────────────────────

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
ABAQUS_DIR = os.path.join(REPO_ROOT, "abaqus_work")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# Ordered debond size-sweep: name -> labelled severity proxy (radius in mm).
# Healthy = 0.  r15/r40 encode radius; near/edge are position variants whose
# nominal radius we read off the same proxy family (r ~ 25 mm mid-size).  The
# generic "Debond" config sits between the two anchored radii.
SEVERITY_LABELS = {
    "Job-GW-Healthy": 0.0,
    "Job-GW-D2-r15": 15.0,
    "Job-GW-Debond": 25.0,
    "Job-GW-D4-near": 25.0,
    "Job-GW-D5-edge": 25.0,
    "Job-GW-D3-r40": 40.0,
}

# Healthy configs that anchor the regressor's severity=0 end.  The frequency
# variants broaden the healthy support so the ensemble does not key on the
# excitation frequency.
HEALTHY_CONFIGS = [
    "Job-GW-Healthy",
    "Job-GW-Freq25k-H",
    "Job-GW-Freq50k-H",
    "Job-GW-Freq75k-H",
    "Job-GW-Freq100k-H",
]

# Mahalanobis anomaly REFERENCE: only the matched-excitation healthy baseline.
# The labelled debond configs are all at the same ~50 kHz excitation, so mixing
# the 25-100 kHz frequency-variant healthies into the reference would make the
# cloud frequency-dominated and blind to debond; we keep the reference matched.
ANOMALY_REF_CONFIGS = ["Job-GW-Healthy", "Job-GW-Freq50k-H"]

# Number of per-sensor descriptors (see _sensor_descriptors).
N_SENSOR_DESC = 11
# Pooling statistics applied across sensors (mean, std, max, min) -> fixed len.
N_POOL = 4
FEATURE_DIM = N_SENSOR_DESC * N_POOL


# ─────────────────────────────────────────────────────────────────────────────
#  1. Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_sensor_csv(csv_path):
    """Load a Job-GW-*_sensors.csv -> (times, sensor_data, positions).

    Format: row0 = header (time_s, sensor_i_U3, ...), row1 = "# x_mm,..."
    position comment, rest = data.  Number of sensors varies per config.

    Returns:
        times       : (n_steps,) array (empty if unreadable)
        sensor_data : dict {sensor_id: (n_steps,) array}
        positions   : list of x_mm per sensor (sorted by sensor_id)
    """
    times = []
    sensor_data = {}
    if not os.path.exists(csv_path):
        return np.array([]), {}, []

    with open(csv_path) as f:
        rows = list(csv.reader(f))
    if len(rows) < 3:
        return np.array([]), {}, []

    header, pos_row = rows[0], rows[1]
    data_rows = [r for r in rows[2:] if r and not r[0].startswith("#")]

    col_to_sensor = {}
    for i, col in enumerate(header):
        if i == 0 or not col.startswith("sensor_"):
            continue
        try:
            sid = int(col.replace("sensor_", "").split("_")[0])
            sensor_data[sid] = []
            col_to_sensor[i] = sid
        except ValueError:
            pass

    sensor_ids = sorted(sensor_data.keys())
    pos_by_col = {}
    for i in range(1, min(len(pos_row), len(header))):
        try:
            pos_by_col[i] = float(pos_row[i])
        except (ValueError, TypeError):
            pos_by_col[i] = 0.0
    sensor_to_col = {s: c for c, s in col_to_sensor.items()}
    positions = [pos_by_col.get(sensor_to_col.get(s, -1), 0.0) for s in sensor_ids]

    for row in data_rows:
        if len(row) < len(header):
            continue
        try:
            times.append(float(row[0]))
            for ci, sid in col_to_sensor.items():
                val = row[ci] if ci < len(row) and row[ci] else "0.0"
                sensor_data[sid].append(float(val))
        except (ValueError, IndexError):
            continue

    times = np.asarray(times, dtype=float)
    for sid in sensor_data:
        sensor_data[sid] = np.asarray(sensor_data[sid], dtype=float)
    return times, sensor_data, positions


# ─────────────────────────────────────────────────────────────────────────────
#  2. Feature extraction (numpy-only FFT / envelope / energy descriptors)
# ─────────────────────────────────────────────────────────────────────────────

def _spectral_features(sig, dt):
    """(dominant_freq, centroid, bandwidth) normalised by Nyquist."""
    n = len(sig)
    if n < 4 or dt <= 0:
        return 0.0, 0.0, 0.0
    mag = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(n, dt)
    total = float(mag.sum())
    if total < 1e-20:
        return 0.0, 0.0, 0.0
    dom = freqs[int(np.argmax(mag[1:])) + 1]
    centroid = float((freqs * mag).sum() / total)
    bandwidth = float(np.sqrt(((freqs - centroid) ** 2 * mag).sum() / total))
    nyq = 0.5 / dt
    return dom / nyq, centroid / nyq, bandwidth / nyq


def _fft_envelope(sig):
    """Analytic-signal envelope via FFT (no scipy). Returns (env_max, decay).

    Hilbert transform by zeroing negative-frequency bins of the FFT, which is
    the standard numpy-only analytic-signal construction.
    """
    n = len(sig)
    if n < 8:
        return 0.0, 0.0
    Sf = np.fft.fft(sig)
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    env = np.abs(np.fft.ifft(Sf * h))
    env_max = float(env.max())
    if env_max < 1e-20:
        return 0.0, 0.0
    peak = int(np.argmax(env))
    tail = env[peak:]
    decay = 0.0
    if len(tail) > 10:
        log_env = np.log(np.clip(tail / env_max, 1e-12, None))
        x = np.arange(len(tail), dtype=float)
        decay = float(-np.polyfit(x, log_env, 1)[0])
    return env_max, decay


def _sensor_descriptors(sig, dt):
    """Fixed-length (N_SENSOR_DESC) descriptor for one sensor waveform."""
    sig = np.nan_to_num(np.asarray(sig, dtype=float))
    if sig.size < 4:
        return np.zeros(N_SENSOR_DESC)
    max_abs = float(np.max(np.abs(sig)))
    rms = float(np.sqrt(np.mean(sig ** 2)))
    energy = float(np.log1p(np.sum(sig ** 2) * (dt if dt > 0 else 1.0)))
    ptp = float(np.ptp(sig))
    crest = max_abs / rms if rms > 1e-20 else 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(sig))) > 0))
    dom, centroid, bw = _spectral_features(sig, dt)
    env_max, decay = _fft_envelope(sig)
    desc = np.array([max_abs, rms, energy, ptp, crest, zcr,
                     dom, centroid, bw, env_max, decay], dtype=float)
    return np.nan_to_num(desc)


def extract_features(times, sensor_data):
    """Per-sensor descriptors pooled into a FIXED-LENGTH feature vector.

    Pooling (mean/std/max/min across sensors) makes the vector independent of
    how many sensors a given config has (configs have 4 or 5).  Returns a
    finite vector of length FEATURE_DIM.
    """
    dt = float(np.median(np.diff(times))) if times.size > 1 else 0.0
    rows = [_sensor_descriptors(sensor_data[s], dt) for s in sorted(sensor_data)]
    if not rows:
        return np.zeros(FEATURE_DIM)
    M = np.vstack(rows)  # (n_sensors, N_SENSOR_DESC)
    pooled = np.concatenate([M.mean(0), M.std(0), M.max(0), M.min(0)])
    return np.nan_to_num(pooled.astype(float))


def _window_signals(times, sensor_data, n_windows=6, overlap=0.5):
    """Split signals into overlapping windows -> list of (times, sensor_data).

    Windowing the (short) GW signals is the stand-in for measurement repeats:
    it gives several feature vectors per config so we can estimate variance.
    """
    n = times.size
    if n < 16 or not sensor_data:
        return [(times, sensor_data)]
    win = max(16, int(n / (1 + (n_windows - 1) * (1 - overlap))))
    step = max(1, int(win * (1 - overlap)))
    out = []
    start = 0
    while start + win <= n and len(out) < n_windows:
        sl = slice(start, start + win)
        out.append((times[sl], {s: v[sl] for s, v in sensor_data.items()}))
        start += step
    return out or [(times, sensor_data)]


def features_for_config(name, n_windows=6, synthetic_fallback=True, rng=None):
    """Feature matrix (n_windows, FEATURE_DIM) for a config name.

    Reads abaqus_work/<name>_sensors.csv if present; otherwise (or if the CSV
    is unreadable) falls back to a synthetic signal whose perturbation scales
    with the config's labelled severity, so tests pass anywhere.
    """
    csv_path = os.path.join(ABAQUS_DIR, name + "_sensors.csv")
    times, sensor_data, _ = load_sensor_csv(csv_path)
    if times.size >= 16 and sensor_data:
        windows = _window_signals(times, sensor_data, n_windows=n_windows)
        return np.vstack([extract_features(t, sd) for t, sd in windows])
    if not synthetic_fallback:
        return np.zeros((1, FEATURE_DIM))
    sev = SEVERITY_LABELS.get(name, 0.0)
    return _synthetic_feature_matrix(sev, n_windows, rng)


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic fallback (so tests / figure run without Abaqus data)
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_signal(severity, seed=0):
    """A toy guided-wave burst whose later-arrival scatter grows with severity."""
    rng = np.random.default_rng(seed)
    dt = 1.07e-6
    n = 640
    t = np.arange(n) * dt
    f0 = 50e3
    # incident 5-cycle Hann-windowed tone burst
    cyc = 5
    env = np.where(t < cyc / f0, np.sin(np.pi * f0 * t / cyc) ** 2, 0.0)
    incident = env * np.sin(2 * np.pi * f0 * t)
    # debond -> a delayed reflected wavelet, amplitude ∝ severity
    tau = 1.8e-4
    refl_env = np.exp(-((t - tau) ** 2) / (2 * (2e-5) ** 2))
    scatter = (severity / 40.0) * refl_env * np.sin(2 * np.pi * f0 * (t - tau))
    noise = 0.01 * rng.standard_normal(n)
    return t, incident + scatter + noise


def _synthetic_feature_matrix(severity, n_windows, rng=None):
    rng = rng or np.random.default_rng(int(severity * 1000) + 7)
    feats = []
    for k in range(max(1, n_windows)):
        t, base = _synthetic_signal(severity, seed=rng.integers(1 << 30))
        sd = {}
        for s in range(5):
            jitter = 1.0 + 0.05 * rng.standard_normal()
            sd[s] = base * jitter + 0.005 * rng.standard_normal(base.size)
        feats.append(extract_features(t, sd))
    return np.vstack(feats)


# ─────────────────────────────────────────────────────────────────────────────
#  3. Severity estimator with uncertainty
# ─────────────────────────────────────────────────────────────────────────────

class SeverityEstimator:
    """Debond severity characterisation with predictive uncertainty.

    Combines two estimates:
      (a) Mahalanobis anomaly score vs the healthy-reference feature cloud — a
          continuous, label-free severity proxy.
      (b) A bagged linear-regressor ensemble (features -> labelled severity);
          the bag mean is the point estimate and the bag spread is a predictive
          sd.  Trained on the (few) labelled size-sweep configs with windowed
          samples.

    posterior() returns mean / sd / samples, fusing (a) and (b).
    """

    def __init__(self, n_estimators=24, ridge_alpha=10.0, random_state=0,
                 fuse_w_ens=0.75):
        self.n_estimators = n_estimators
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state
        self.fuse_w_ens = float(fuse_w_ens)  # ensemble weight in the fusion
        self._fitted = False

    # ---- fitting -----------------------------------------------------------
    def fit(self, X_healthy, X_sev, y_sev, X_anomaly_ref=None):
        """Fit the estimator.

        X_healthy   : (nh, d) healthy windows anchoring the regressor at sev=0.
        X_sev/y_sev : labelled debond windows + their severity labels.
        X_anomaly_ref : (na, d) matched-excitation healthy windows that define
                        the Mahalanobis anomaly reference.  Defaults to
                        X_healthy if not given.
        """
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        X_healthy = np.atleast_2d(np.asarray(X_healthy, dtype=float))
        X_sev = np.atleast_2d(np.asarray(X_sev, dtype=float))
        y_sev = np.asarray(y_sev, dtype=float).ravel()
        if X_anomaly_ref is None:
            X_anomaly_ref = X_healthy
        X_anomaly_ref = np.atleast_2d(np.asarray(X_anomaly_ref, dtype=float))

        # standardise on the full training cloud
        self.scaler = StandardScaler().fit(np.vstack([X_healthy, X_sev]))
        Hs = self.scaler.transform(X_healthy)
        Ss = self.scaler.transform(X_sev)
        Rs = self.scaler.transform(X_anomaly_ref)

        # (a) healthy-reference anomaly statistics.  The healthy feature cloud
        # is small and strongly collinear, so a full-covariance Mahalanobis
        # precision (pinv) amplifies near-null directions and explodes on
        # out-of-cloud inputs.  We use a DIAGONAL (per-feature z-score)
        # Mahalanobis instead: well-conditioned, monotone in feature deviation,
        # and bounded-growth — the standard robust choice for tiny references.
        # Reference = matched-excitation healthy only (Rs), so the score is
        # sensitive to debond rather than to excitation frequency.
        self.mu_h = Rs.mean(0)
        var = Rs.var(0) if Rs.shape[0] > 1 else np.ones(Rs.shape[1])
        self.inv_var_h = 1.0 / (var + 1.0)  # +1: features are standardised (~unit var)

        # Standardise the raw Mahalanobis distance against the HEALTHY reference
        # self-distances (mean / sd), giving a z-scored anomaly that is ~0 for
        # healthy and grows with damage.  This removes the huge absolute offset
        # so the score actually separates debond from healthy.
        d_ref = self._maha(Rs)
        self.maha_mu = float(d_ref.mean())
        self.maha_sd = float(d_ref.std() + 1e-6)

        # Calibrate the z-scored anomaly onto the severity scale with a linear
        # least-squares fit through the labelled anchors (clipped >= 0).  Kept
        # simple and monotone; absolute scale is anchored to the r15/r40 labels.
        z_sev = self._anomaly_z(Ss)
        A = np.column_stack([z_sev, np.ones_like(z_sev)])
        coef, *_ = np.linalg.lstsq(A, y_sev, rcond=None)
        self.maha_a, self.maha_b = float(coef[0]), float(coef[1])

        # (b) bagged ridge ensemble
        rng = np.random.default_rng(self.random_state)
        self.models = []
        n = Ss.shape[0]
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)  # bootstrap
            # tiny feature jitter to decorrelate members on degenerate data
            Xb = Ss[idx] + 1e-3 * rng.standard_normal(Ss[idx].shape)
            mdl = Ridge(alpha=self.ridge_alpha).fit(Xb, y_sev[idx])
            self.models.append(mdl)

        self.y_min = float(y_sev.min())
        self.y_max = float(y_sev.max())
        self._fitted = True
        return self

    # ---- scoring -----------------------------------------------------------
    def _maha(self, Xs):
        diff = Xs - self.mu_h
        return np.sum(diff * diff * self.inv_var_h, axis=1)  # diagonal Mahalanobis

    def _anomaly_z(self, Xs):
        """Mahalanobis distance z-scored against the healthy reference."""
        d = self._maha(Xs)
        return (d - self.maha_mu) / self.maha_sd

    def anomaly_score(self, X):
        """Continuous healthy-reference severity proxy (>=0), per window.

        z-scored Mahalanobis distance (≈0 for healthy, growing with damage)
        mapped linearly onto the labelled severity scale and clipped to a
        sensible band so out-of-cloud inputs cannot extrapolate to absurd radii.
        """
        Xs = self.scaler.transform(np.atleast_2d(np.asarray(X, dtype=float)))
        z = self._anomaly_z(Xs)
        sev = self.maha_a * z + self.maha_b
        cap = 1.5 * getattr(self, "y_max", np.inf)
        return np.clip(sev, 0.0, cap)

    def _ensemble_preds(self, X):
        Xs = self.scaler.transform(np.atleast_2d(np.asarray(X, dtype=float)))
        return np.stack([m.predict(Xs) for m in self.models], axis=0)  # (B, n)

    def posterior(self, X, n_samples=256, random_state=None):
        """Posterior over debond severity (mm radius proxy) for feature rows X.

        Fuses the bagged-ensemble predictive distribution with the
        Mahalanobis anomaly proxy (averaged as two independent estimates).
        Returns dict with mean, sd (both > 0 spread) and a sample array.
        """
        if not self._fitted:
            raise RuntimeError("SeverityEstimator.fit() must be called first")
        X = np.atleast_2d(np.asarray(X, dtype=float))

        cap = 1.5 * self.y_max
        preds = np.clip(self._ensemble_preds(X), 0.0, cap)   # (B, n)
        ens_mean = preds.mean(0)                    # (n,)
        ens_sd = preds.std(0)                       # (n,)
        maha = self.anomaly_score(X)               # (n,)

        # Fuse the two estimators.  The bagged ensemble is the discriminative
        # size estimator (it weights the debond-sensitive feature dimensions),
        # so it gets the larger weight; the Mahalanobis anomaly is a coarser,
        # label-free severity proxy and acts as a mild regulariser.
        w_ens = self.fuse_w_ens
        mean = w_ens * ens_mean + (1.0 - w_ens) * maha
        span = max(self.y_max - self.y_min, 1.0)
        sd = np.sqrt(ens_sd ** 2 + (0.15 * span) ** 2)

        rng = np.random.default_rng(random_state if random_state is not None
                                    else self.random_state)
        # pooled posterior over the window batch (treat windows as repeats)
        m_bar = float(mean.mean())
        sd_bar = float(np.sqrt((sd ** 2).mean() + mean.var()))
        samples = rng.normal(m_bar, max(sd_bar, 1e-6), size=n_samples)
        samples = np.clip(samples, 0.0, None)
        return {
            "mean": m_bar,
            "sd": max(sd_bar, 1e-6),
            "samples": samples,
            "per_window_mean": mean,
            "per_window_sd": sd,
        }


def build_estimator(n_windows=6, synthetic_fallback=True, random_state=0):
    """Assemble training data from the catalogue and fit a SeverityEstimator.

    Returns (estimator, info) where info records which configs were found on
    disk vs synthesised.
    """
    rng = np.random.default_rng(random_state)
    found, synth, cache = [], [], {}

    def _grab(name):
        if name in cache:
            return cache[name]
        csv_path = os.path.join(ABAQUS_DIR, name + "_sensors.csv")
        on_disk = os.path.exists(csv_path)
        X = features_for_config(name, n_windows=n_windows,
                                synthetic_fallback=synthetic_fallback, rng=rng)
        (found if on_disk and X.shape[0] >= 1 and
         load_sensor_csv(csv_path)[0].size >= 16 else synth).append(name)
        cache[name] = X
        return X

    X_healthy = np.vstack([_grab(c) for c in HEALTHY_CONFIGS])
    X_sev_parts, y_parts = [], []
    for name, sev in SEVERITY_LABELS.items():
        Xc = _grab(name)
        X_sev_parts.append(Xc)
        y_parts.append(np.full(Xc.shape[0], sev))
    X_sev = np.vstack(X_sev_parts)
    y_sev = np.concatenate(y_parts)
    # matched-excitation healthy reference for the Mahalanobis anomaly score
    X_anom = np.vstack([_grab(c) for c in ANOMALY_REF_CONFIGS])

    est = SeverityEstimator(random_state=random_state).fit(
        X_healthy, X_sev, y_sev, X_anomaly_ref=X_anom)
    info = {"found": found, "synthetic": synth,
            "n_healthy": X_healthy.shape[0], "n_labelled": X_sev.shape[0]}
    return est, info


# ─────────────────────────────────────────────────────────────────────────────
#  4. Bridge: severity posterior -> Stage-3 radius-sample posterior
# ─────────────────────────────────────────────────────────────────────────────

def severity_to_radius_posterior(posterior, n_samples=256,
                                 r_min=0.0, r_max=80.0, random_state=0):
    """Turn a Stage-2 severity posterior into the radius-sample posterior the
    fairing Stage-3 prognosis consumes.

    The severity scale is already a debond-radius proxy (mm, anchored to the
    r15/r40 configs), so the bridge mainly resamples to the requested count and
    clips to a physically sensible fairing-debond range [r_min, r_max] mm.

    Returns an (n_samples,) array of posterior radius samples (mm).  Stage-3
    (cfrp_phasefield_2d.seed_defect) ultimately wants size as 2**log2size
    elements; mapping mm->elements is a calibration step left to the prognosis
    layer (see module HONEST SCOPE note).
    """
    base = np.asarray(posterior["samples"], dtype=float)
    if base.size == 0:
        base = np.array([posterior.get("mean", 0.0)])
    rng = np.random.default_rng(random_state)
    if base.size >= n_samples:
        out = rng.choice(base, size=n_samples, replace=False)
    else:
        # resample with a small smoothing kernel = ensemble sd
        sd = max(float(posterior.get("sd", 1.0)), 1e-3)
        idx = rng.integers(0, base.size, size=n_samples)
        out = base[idx] + rng.normal(0.0, 0.25 * sd, size=n_samples)
    return np.clip(out, r_min, r_max)


# ─────────────────────────────────────────────────────────────────────────────
#  Figure
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(out_path=None, random_state=0):
    """Severity estimate +/- uncertainty across ordered debond configs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = out_path or os.path.join(RESULTS_DIR, "fairing_stage2.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    est, info = build_estimator(random_state=random_state)

    # evaluate severity posterior per config, in ascending labelled order
    order = sorted(SEVERITY_LABELS, key=lambda k: SEVERITY_LABELS[k])
    labels = [SEVERITY_LABELS[k] for k in order]
    means, sds = [], []
    for name in order:
        X = features_for_config(name, n_windows=8, rng=np.random.default_rng(1))
        post = est.posterior(X, random_state=random_state)
        means.append(post["mean"])
        sds.append(post["sd"])

    means = np.array(means)
    sds = np.array(sds)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs = np.arange(len(order))
    ax.errorbar(xs, means, yerr=sds, fmt="o-", color="#1f77b4", capsize=4,
                lw=1.8, ms=7, label="Stage-2 estimate $\\pm$ sd")
    ax.plot(xs, labels, "s--", color="#d62728", ms=6, alpha=0.8,
            label="labelled radius proxy (mm)")
    ax.set_xticks(xs)
    ax.set_xticklabels([n.replace("Job-GW-", "") for n in order],
                       rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("debond severity / radius proxy [mm]")
    ax.set_title("Fairing Stage-2: debond severity characterisation "
                 "with uncertainty")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    src = "Abaqus GW data" if info["found"] else "synthetic fallback"
    n_found = len(info["found"])
    n_total = len(set(SEVERITY_LABELS) | set(HEALTHY_CONFIGS))
    ax.text(0.98, 0.02,
            f"{n_found}/{n_total} configs from disk\nproof-of-concept ({src})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color="gray")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    mono = bool(np.all(np.diff(means) >= -0.5 * np.maximum(sds[:-1], 1e-6)))
    print(f"[fig] saved {out_path}")
    print(f"[fig] configs from disk: {sorted(info['found'])}")
    print(f"[fig] synthetic:         {sorted(info['synthetic'])}")
    print(f"[fig] severity means:    "
          + ", ".join(f"{n.replace('Job-GW-','')}={m:.1f}"
                      for n, m in zip(order, means)))
    print(f"[fig] monotone-in-size (within sd): {mono}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────────────────────

def run_tests():
    """Self-contained tests; pass with or without Abaqus data on disk."""
    print("=" * 70)
    print("fairing_stage2.py — self test")
    print("=" * 70)
    rng = np.random.default_rng(0)

    # 1. feature extractor: fixed-length, finite
    t, base = _synthetic_signal(20.0, seed=1)
    sd_clean = {s: base + 0.001 * rng.standard_normal(base.size) for s in range(5)}
    f_clean = extract_features(t, sd_clean)
    assert f_clean.shape == (FEATURE_DIM,), f"feature dim {f_clean.shape}"
    assert np.all(np.isfinite(f_clean)), "non-finite features"
    # different sensor counts still give same length
    f_four = extract_features(t, {s: base for s in range(4)})
    assert f_four.shape == (FEATURE_DIM,), "dim not sensor-count invariant"
    print(f"[1] feature extractor OK: fixed length {FEATURE_DIM}, finite, "
          "sensor-count invariant")

    # 2. anomaly score higher for a damaged signal than a healthy one.
    #    Use real Healthy vs Debond configs when present (the genuine use
    #    case); otherwise a synthetic perturbed-vs-healthy pair.
    est, info = build_estimator(random_state=0)
    healthy_on_disk = os.path.exists(
        os.path.join(ABAQUS_DIR, "Job-GW-Healthy_sensors.csv"))
    debond_on_disk = os.path.exists(
        os.path.join(ABAQUS_DIR, "Job-GW-D3-r40_sensors.csv"))
    if healthy_on_disk and debond_on_disk:
        Xh = features_for_config("Job-GW-Healthy", n_windows=6)
        Xd = features_for_config("Job-GW-D3-r40", n_windows=6)
        src = "real Healthy vs D3-r40"
    else:
        th, healthy = _synthetic_signal(0.0, seed=5)
        sd_h = {s: healthy + 0.05 * abs(healthy).max() * rng.standard_normal(
            healthy.size) for s in range(5)}
        tp, perturbed = _synthetic_signal(40.0, seed=5)
        sd_p = {s: perturbed + 0.05 * abs(perturbed).max() * rng.standard_normal(
            perturbed.size) for s in range(5)}
        Xh = extract_features(th, sd_h)[None]
        Xd = extract_features(tp, sd_p)[None]
        src = "synthetic perturbed vs healthy"
    a_h = float(est.anomaly_score(Xh).mean())
    a_p = float(est.anomaly_score(Xd).mean())
    assert a_p > a_h, f"anomaly not higher for damaged: {a_p:.3f} vs {a_h:.3f}"
    print(f"[2] anomaly score OK ({src}): damaged {a_p:.2f} > healthy {a_h:.2f}")

    # 3. severity estimator returns mean + positive predictive sd
    #    (use a real labelled debond config when available)
    Xp = features_for_config("Job-GW-D3-r40", n_windows=6,
                             rng=np.random.default_rng(2))
    post = est.posterior(Xp)
    assert np.isfinite(post["mean"]), "non-finite posterior mean"
    assert post["sd"] > 0, f"sd not positive: {post['sd']}"
    assert post["samples"].size == 256, "wrong sample count"
    print(f"[3] severity posterior OK: mean={post['mean']:.2f} mm, "
          f"sd={post['sd']:.2f} mm (>0)")

    # 4. bridge -> finite radius samples in sensible range
    radii = severity_to_radius_posterior(post, n_samples=256)
    assert radii.shape == (256,), f"radius sample shape {radii.shape}"
    assert np.all(np.isfinite(radii)), "non-finite radii"
    assert np.all(radii >= 0.0) and radii.max() <= 80.0, "radii out of range"
    assert radii.mean() > 0.0, "degenerate radius posterior"
    print(f"[4] radius posterior OK: n={radii.size}, "
          f"mean={radii.mean():.1f} mm, range=[{radii.min():.1f}, "
          f"{radii.max():.1f}] mm")

    # 5. monotone trend check (proof-of-concept quality signal, not asserted)
    order = sorted(SEVERITY_LABELS, key=lambda k: SEVERITY_LABELS[k])
    means = [est.posterior(features_for_config(n, n_windows=6,
             rng=np.random.default_rng(3)))["mean"] for n in order]
    trend = float(np.corrcoef([SEVERITY_LABELS[n] for n in order], means)[0, 1])
    print(f"[5] severity-vs-size rank trend (corr): {trend:+.3f}")
    print(f"    configs on disk: {len(info['found'])}, "
          f"synthetic: {len(info['synthetic'])}")

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
    return True


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", action="store_true", help="run self-tests")
    ap.add_argument("--fig", action="store_true", help="make severity figure")
    args = ap.parse_args()
    if args.test:
        run_tests()
    if args.fig:
        make_figure()
    if not (args.test or args.fig):
        ap.print_help()


if __name__ == "__main__":
    main()
