import os, json, math, time, random, glob
from datetime import datetime
from collections import defaultdict
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HAS_SK = False
HAS_STATSMODELS = False
HAS_IDTXL = False

try:
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,roc_curve
    from scipy.stats import zscore as _zscore_scipy
    HAS_SK = True
except Exception:
    pass

try:
    from statsmodels.api import OLS, add_constant
    HAS_STATSMODELS = True
except Exception:
    pass

try:
    from IDTxl.idtxl.estimator import get_estimator
    HAS_IDTXL = True
except Exception:
    HAS_IDTXL = False


DATA_DIR = os.environ.get("KRL_DATA_DIR", "files/wolves")
GLOB_PATTERN = "simulation-wolves-*.csv"
OUT_DIR = os.environ.get("KRL_OUT_DIR", "results_krl_out_wolves")
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)
CSV_DIR = os.path.join(OUT_DIR, "csv")
os.makedirs(CSV_DIR, exist_ok=True)

ALPHA_ID = int(os.environ.get("KRL_ALPHA_ID", "100"))
FOLLOWERS = set(range(101, 115))
INDEPENDENTS = set(range(115, 130))

WINDOW_SIZE = int(os.environ.get("KRL_WINDOW_SIZE", "200"))
WINDOW_STEP = int(os.environ.get("KRL_WINDOW_STEP", "100"))
TOPK_PER_TGT = int(os.environ.get("KRL_TOPK", "8"))
N_SURROGATES = int(os.environ.get("KRL_SURR", "30"))
BOOTSTRAP_B = int(os.environ.get("KRL_BOOT_B", "500"))

STRIDES = [1, 2, 3]
LAGS = [1, 2, 3, 4]
TE_NBINS = [6, 8, 12]
GC_MAXLAGS = [2, 4, 6]
DIST_THRESHOLDS = [np.inf, 50.0, 25.0, 15.0]

LAG_TEMP = float(os.environ.get("KRL_LAG_TEMP", "0.5"))
RIDGE_ALPHA = float(os.environ.get("KRL_RIDGE_ALPHA", "1.0"))
MAIN_K = int(os.environ.get("KRL_MAIN_K", "30"))

TAU_DIST = float(os.environ.get("KRL_TAU_DIST", "35.0"))
TAU_ALIGN = float(os.environ.get("KRL_TAU_ALIGN", "0.25"))
TAU_R2 = float(os.environ.get("KRL_TAU_R2", "0.01"))
TAU_LCONS = float(os.environ.get("KRL_TAU_LCONS", "0.4"))

USE_CALIBRATION = os.environ.get("KRL_USE_CALIB", "1") == "1"
RANDOM_SEED = 7
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

HP_SWEEP_FLAG = os.environ.get("KRL_HP_SWEEP", "0") == "1"

HP_GRID = [
    {
        "name": "hp_low_gate",
        "LAG_TEMP": 0.3,
        "TAU_DIST": TAU_DIST * 0.8,
        "TAU_ALIGN": TAU_ALIGN * 0.8,
        "TAU_R2": TAU_R2,
        "TAU_LCONS": TAU_LCONS * 0.8,
    },
    {
        "name": "hp_default",
        "LAG_TEMP": LAG_TEMP,
        "TAU_DIST": TAU_DIST,
        "TAU_ALIGN": TAU_ALIGN,
        "TAU_R2": TAU_R2,
        "TAU_LCONS": TAU_LCONS,
    },
    {
        "name": "hp_high_gate",
        "LAG_TEMP": 1.0,
        "TAU_DIST": TAU_DIST * 1.2,
        "TAU_ALIGN": TAU_ALIGN * 1.2,
        "TAU_R2": TAU_R2,
        "TAU_LCONS": TAU_LCONS * 1.2,
    },
]

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def winsorize(x, p=0.01):
    x = np.asarray(x, float)
    if len(x) == 0 or np.all(np.isnan(x)):
        return x
    lo = np.nanpercentile(x, 100 * p)
    hi = np.nanpercentile(x, 100 * (1 - p))
    return np.clip(x, lo, hi)


def zscore(v):
    v = np.asarray(v, float)
    if np.all(np.isnan(v)):
        return np.zeros_like(v)
    v = winsorize(v, 0.01)
    mu, sd = np.nanmean(v), np.nanstd(v)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(v)
    return (v - mu) / (sd + 1e-12)


def cliffs_delta(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = sum(np.sum(xi > y) for xi in x)
    lt = sum(np.sum(xi < y) for xi in x)
    return float((gt - lt) / (len(x) * len(y) + 1e-9))


def roc_pr_curves(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    m = np.isfinite(s)
    if m.sum() < 2 or len(np.unique(y[m])) < 2:
        return (
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.5]),
            np.nan,
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.nan,
        )
    fpr, tpr, thr_roc = roc_curve(y[m], s[m])
    try:
        auc = roc_auc_score(y[m], s[m])
    except Exception:
        auc = np.nan
    prec, rec, thr_pr = precision_recall_curve(y[m], s[m])
    try:
        aupr = average_precision_score(y[m], s[m])
    except Exception:
        aupr = np.nan
    return fpr, tpr, thr_roc, auc, prec, rec, thr_pr, aupr


def gate_weighted_stat(scores, gates):
    scores = np.asarray(scores, float)
    gates = np.asarray(gates, float)
    m = np.isfinite(scores) & np.isfinite(gates)
    if m.sum() == 0:
        return np.nan
    return float(np.nanmean(scores[m] * np.clip(gates[m], 0.0, 1.0)))


def compute_features(df):
    df = df.sort_values(["id_agent", "time"]).copy()
    df["dx"] = df.groupby("id_agent")["coordinate_x"].diff()
    df["dy"] = df.groupby("id_agent")["coordinate_y"].diff()
    df["heading"] = (
        np.mod(np.arctan2(df["dy"], df["dx"]) + np.pi, 2 * np.pi) - np.pi
    )
    df["turn_rate"] = df.groupby("id_agent")["heading"].diff()
    df["turn_rate"] = np.mod(df["turn_rate"] + np.pi, 2 * np.pi) - np.pi
    if "speed" in df.columns:
        df["accel"] = df.groupby("id_agent")["speed"].diff()
    else:
        df["speed"] = np.nan
        df["accel"] = np.nan
    return df


def window_indices(T, win, step):
    start = 0
    while start < T:
        end = min(start + win, T)
        yield start, end
        if end == T:
            break
        start += step


def pairwise_mean_distance(A, B):
    n = min(len(A), len(B))
    if n == 0:
        return np.nan
    return float(np.linalg.norm(A[:n] - B[:n], axis=1).mean())


def median_distance_window(A, B, w0, w1):
    a = A[w0:w1]
    b = B[w0:w1]
    n = min(len(a), len(b))
    if n == 0:
        return np.nan
    return float(np.median(np.linalg.norm(a[:n] - b[:n], axis=1)))


def knn_alignment(src_head, tgt_head, k=5):
    n = min(len(src_head), len(tgt_head))
    if n == 0:
        return np.nan
    sx, sy = np.cos(src_head[:n]), np.sin(src_head[:n])
    tx, ty = np.cos(tgt_head[:n]), np.sin(tgt_head[:n])
    w = np.ones(k) / k
    Sx, Sy = np.convolve(sx, w, "same"), np.convolve(sy, w, "same")
    Tx, Ty = np.convolve(tx, w, "same"), np.convolve(ty, w, "same")
    num = Sx * Tx + Sy * Ty
    den = np.hypot(Sx, Sy) * np.hypot(Tx, Ty) + 1e-9
    return float(np.nanmean(num / den))


def dtw_distance_xy(A, B, radius=5):
    if len(A) == 0 or len(B) == 0:
        return np.nan
    ax, ay = A[:, 0], A[:, 1]
    bx, by = B[:, 0], B[:, 1]
    n, m = len(ax), len(bx)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j0, j1 = max(1, i - radius), min(m, i + radius)
        for j in range(j0, j1 + 1):
            cost = (ax[i - 1] - bx[j - 1]) ** 2 + (ay[i - 1] - by[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(math.sqrt(D[n, m] / max(n, m)))


def _build_auto_target_matrix(tgt_head):
    y_t = np.asarray(tgt_head, float)
    y_tm1 = np.roll(y_t, 1)
    y_tm1[0] = np.nan
    X_auto = np.column_stack([np.sin(y_tm1), np.cos(y_tm1)])
    return X_auto, y_t


def _build_src_kinematics(src_head, src_turn, src_speed, src_accel, lag):
    sh = pd.Series(src_head).shift(lag).values
    tr = pd.Series(src_turn).shift(lag).values
    sp = pd.Series(src_speed).shift(lag).values
    ac = pd.Series(src_accel).shift(lag).values
    Xk = np.column_stack(
        [
            np.sin(sh),
            np.cos(sh),
            np.sin(tr),
            np.cos(tr),
            zscore(sp),
            zscore(ac)
        ]
    )
    return Xk


def _build_relational(src_df, tgt_df, lag):
    sh = pd.Series(src_df["heading"].values).shift(lag).values
    sp = pd.Series(src_df["speed"].values).shift(lag).values
    sx = pd.Series(src_df["coordinate_x"].values).shift(lag).values
    sy = pd.Series(src_df["coordinate_y"].values).shift(lag).values

    th = tgt_df["heading"].values
    tx = tgt_df["coordinate_x"].values
    ty = tgt_df["coordinate_y"].values
    tv = tgt_df["speed"].values

    dx, dy = tx - sx, ty - sy
    dist = np.sqrt(dx**2 + dy**2)
    bearing = np.arctan2(dy, dx)
    align = np.cos(th - sh)
    dv = tv - sp
    ratio = np.divide(tv, np.where(np.abs(sp) < 1e-6, np.nan, sp))

    Xr = np.column_stack(
        [zscore(dist), np.sin(bearing), np.cos(bearing), align, zscore(dv), zscore(ratio)]
    )
    return Xr


def _fit_ridge(X, y, alpha=RIDGE_ALPHA):
    try:
        m = Ridge(alpha=alpha, fit_intercept=True)
        m.fit(X, y)
        return m, m.predict(X)
    except Exception:
        return None, None


def krl_components(src, tgt, lags, temp=LAG_TEMP):
    if not HAS_SK:
        return np.nan, [], [], {}

    tgt_head = tgt["heading"].values
    src_head = src["heading"].values
    src_turn = src["turn_rate"].values
    src_speed = src["speed"].values
    src_acc = src["accel"].values

    Xa, y = _build_auto_target_matrix(tgt_head)

    m0 = ~np.isnan(y) & ~np.isnan(Xa).any(1)
    if m0.sum() < 30:
        return np.nan, [], [], {}

    m_a, yhat_a = _fit_ridge(Xa[m0], y[m0])
    if yhat_a is None:
        return np.nan, [], [], {}

    ss_res_a = float(np.sum((y[m0] - yhat_a) ** 2))
    ss_tot = float(np.sum((y[m0] - np.nanmean(y[m0])) ** 2) + 1e-9)
    r2_auto = max(0.0, 1.0 - ss_res_a / ss_tot)

    gains = []
    for lag in lags:
        Xk = _build_src_kinematics(src_head, src_turn, src_speed, src_acc, lag)
        Xr = _build_relational(src, tgt, lag)
        X = np.column_stack([Xa, Xk, Xr]).astype(float)
        m = ~np.isnan(y) & ~np.isnan(X).any(1)
        if m.sum() < 30:
            gains.append(np.nan)
            continue
        _, yhat = _fit_ridge(X[m], y[m])
        if yhat is None:
            gains.append(np.nan)
            continue
        ss_res = float(np.sum((y[m] - yhat) ** 2))
        r2_full = max(0.0, 1.0 - ss_res / ss_tot)
        gains.append(max(0.0, r2_full - r2_auto))

    g = np.nan_to_num(np.array(gains, float), nan=0.0)
    if np.all(g == 0):
        w = np.zeros_like(g)
    else:
        w = np.exp(g / max(1e-6, temp))
        w /= w.sum() + 1e-12

    lag_cons = (
        float(np.sort(w)[-2:].sum())
        if w.size > 1
        else float(w.max() if w.size else np.nan)
    )
    return r2_auto, list(g), list(w), {
        "r2_auto": r2_auto,
        "r2_gain_max": float(np.nanmax(g) if len(g) else np.nan),
        "lag_consistency": lag_cons
    }


def compute_gate(dist_med, align_mean, r2_gain_max, lag_consistency):
    x = np.array(
        [
            (TAU_DIST - dist_med) / max(1.0, TAU_DIST),
            (align_mean - TAU_ALIGN) / max(1e-6, abs(TAU_ALIGN)),
            (r2_gain_max - TAU_R2) / max(1e-6, abs(TAU_R2)),
            (lag_consistency - TAU_LCONS) / max(1e-6, abs(TAU_LCONS))
        ],
        float
    )
    w = np.array([0.6, 1.2, 1.0, 1.0])
    z = float(np.dot(w, x))
    g = 1.0 / (1.0 + math.exp(-z))
    return float(np.clip(g, 0.0, 1.0))


def tsmi_core(src_head, tgt_head, lag, weights=None, stride=1):
    if not HAS_SK:
        return np.nan
    y_lag = pd.Series(tgt_head).shift(-lag).values
    xs = np.sin(src_head)[::stride]
    xc = np.cos(src_head)[::stride]
    yl = y_lag[::stride]
    m = ~np.isnan(xs) & ~np.isnan(xc) & ~np.isnan(yl)
    if m.sum() < 10:
        return np.nan
    X = np.column_stack([xs[m], xc[m]])
    Y = yl[m]
    if weights is not None:
        w = np.asarray(weights, float)[::stride][m]
        w = np.nan_to_num(w, nan=0.0)
    else:
        w = None
    try:
        if w is None or (w <= 0).sum() == len(w):
            beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            Yh = X @ beta
        else:
            lr = LinearRegression()
            lr.fit(X, Y, sample_weight=w)
            Yh = lr.predict(X)
        ss_res = np.sum((Y - Yh) ** 2)
        ss_tot = np.sum((Y - Y.mean()) ** 2) + 1e-9
        return float(1.0 - ss_res / ss_tot)
    except Exception:
        return np.nan

def tsmi_over_lags(src_head, tgt_head, lag_grid, weights=None, stride=1, lag_weights=None):
    lag_grid = list(lag_grid)
    if lag_weights is None:
        lag_weights = np.ones(len(lag_grid), float)
    else:
        lag_weights = np.asarray(lag_weights, float)
        if lag_weights.shape[0] != len(lag_grid):
            lag_weights = np.ones(len(lag_grid), float)

    vals = []
    ws = []
    for lag, w_lag in zip(lag_grid, lag_weights):
        if w_lag <= 0:
            continue
        v = tsmi_core(src_head, tgt_head, lag=lag, weights=weights, stride=stride)
        if np.isfinite(v):
            vals.append(w_lag * v)
            ws.append(abs(w_lag))

    if len(vals) == 0:
        return np.nan
    return float(np.nansum(vals) / (np.sum(ws) + 1e-12))

def te_core_partial_idtxl(
    src_sig,
    tgt_sig,
    lag,
    z_feat,
    nbins=8,
    use_z=False,
    max_z_dims=2
):
    if not HAS_IDTXL:
        return np.nan

    y_t = pd.Series(tgt_sig).values   
    y_tm1 = pd.Series(tgt_sig).shift(1).values
    x_l = pd.Series(src_sig).shift(lag).values

    z_feat = np.asarray(z_feat, float)
    if z_feat.ndim == 1:
        z_feat = z_feat.reshape(-1, 1)

    if use_z and z_feat.size > 0:
        z_feat = (z_feat - np.nanmean(z_feat, axis=0)) / (np.nanstd(z_feat, axis=0) + 1e-12)
        d = min(max_z_dims, z_feat.shape[1])
        if d > 0:
            pca = PCA(n_components=d)
            mask_z = ~np.isnan(z_feat).any(axis=1)
            if mask_z.sum() > d + 10:
                z_clean = z_feat[mask_z]
                z_red = np.full((len(z_feat), d), np.nan, float)
                z_red[mask_z] = pca.fit_transform(z_clean)
                z_feat = z_red
            else:
                use_z = False

    if use_z and z_feat.size > 0:
        M = np.column_stack([y_t, y_tm1, x_l, z_feat])
    else:
        M = np.column_stack([y_t, y_tm1, x_l])

    mask = ~np.isnan(M).any(axis=1)
    M = M[mask]

    if M.shape[0] < 150:
        return np.nan

    Y = M[:, 0:1]
    Xl = M[:, 2:3]

    if use_z and z_feat.size > 0:
        cond = np.column_stack([M[:, 1:2], M[:, 3:]])
    else:
        cond = M[:, 1:2]

    try:
        settings = {
            "history_target": 1,
            "history_source": 1,
            "tau_target": 1,
            "tau_source": 1,
            "source_target_delay": 1,
            "kraskov_k": 3, 
            "normalise": False,
            "theiler_t": 0,
            "noise_level": 1e-8,
            "num_threads": "USE_ALL",
            "debug": False,
            "local_values": False,
        }

        est = get_estimator("JidtKraskovCMI", settings)
        te_val = est.estimate(var1=Y, var2=Xl, conditional=cond)

        te_val = float(te_val)
        if te_val < 0:
            te_val = 0.0

        print("[DEBUG TE-Kraskov] ok  N=", M.shape[0], " TE=", te_val)
        return te_val

    except Exception as e:
        print("[TE ERROR] IDTxL Kraskov failed:", repr(e))
        print("    shapes: Y", np.shape(Y),
              " Xl", np.shape(Xl),
              " cond", np.shape(cond))
        return np.nan


def select_te_nbins_from_signal(y_values, te_nbins_grid=TE_NBINS):
    vals = np.asarray(y_values, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return te_nbins_grid[0]

    nb_counts = []
    for b in te_nbins_grid:
        if vals.size <= max(b, 5):
            nb_counts.append(0)
            continue
        try:
            bins = pd.qcut(vals, b, duplicates="drop")
            nb_counts.append(len(np.unique(bins)))
        except Exception:
            nb_counts.append(0)

    if sum(nb_counts) == 0:
        return te_nbins_grid[0]

    best_idx = int(np.argmax(nb_counts))
    return te_nbins_grid[best_idx]

def gc_core_arx(src_sig, tgt_sig, lag, z_feat, maxlag=4, weights=None):
    if not HAS_STATSMODELS:
        return np.nan
    y = pd.Series(tgt_sig).values
    ytm1 = pd.Series(tgt_sig).shift(1).values
    xlag = pd.Series(src_sig).shift(lag).values
    M = np.column_stack([ytm1, xlag, z_feat])
    mask = ~np.isnan(M).any(1) & ~np.isnan(y)
    if mask.sum() < maxlag + 20:
        return np.nan
    y = y[mask]
    Xfull = M[mask]
    X0 = np.column_stack([Xfull[:, 0], Xfull[:, 2:]])
    X1 = Xfull
    try:
        if weights is not None:
            w = np.nan_to_num(np.asarray(weights, float)[mask], nan=0.0)
            W = np.sqrt(w)
            y0, X0w = y * W, X0 * W[:, None]
            y1, X1w = y * W, X1 * W[:, None]
        else:
            y0, X0w = y, X0
            y1, X1w = y, X1
        m0 = OLS(y0, add_constant(X0w)).fit()
        m1 = OLS(y1, add_constant(X1w)).fit()
        df1 = m1.df_model - m0.df_model
        df2 = m1.df_resid
        if df1 <= 0 or df2 <= 0:
            return np.nan
        RSS0, RSS1 = m0.ssr, m1.ssr
        F = ((RSS0 - RSS1) / df1) / (RSS1 / df2 + 1e-12)
        try:
            from scipy.stats import f as fdist

            p = 1.0 - fdist.cdf(F, df1, df2)
        except Exception:
            p = np.exp(-F)
        return float(1.0 - min(max(p, 0.0), 1.0))
    except Exception:
        return np.nan


def load_all_simulations(data_dir, pattern):
    files = sorted(glob.glob(os.path.join(data_dir, pattern)),
        key=lambda x: int(os.path.basename(x).split("-")[-1].split(".")[0])
    )
    if not files:
        raise FileNotFoundError("No CSV found.")
    dfs = []
    for f in files:
        sim_id = int(os.path.basename(f).split("-")[-1].split(".")[0])
        df = pd.read_csv(f)
        df.columns = [c.lower() for c in df.columns]
        req = {"coordinate_x", "coordinate_y", "id_agent", "time", "speed", "label"}
        miss = req - set(df.columns)
        if miss:
            raise ValueError(f"{f} missing columns {miss}")
        df["simulation"] = sim_id
        dfs.append(df)
    return (pd.concat(dfs, ignore_index=True).sort_values(["simulation", "time", "id_agent"]).reset_index(drop=True))


def _choose_gc_maxlag(sh, th, gc_maxlag_grid):
    v = np.nanvar(sh - th)
    if not np.isfinite(v):
        v = 0.0
    idx = int(round(v))
    idx = max(0, min(len(gc_maxlag_grid) - 1, idx))
    return gc_maxlag_grid[idx]


def compute_edge_scores_window_core(
    df_sim,
    lag_grid,
    stride,
    nbins_te_grid,
    gc_maxlag_grid,
    dist_threshold,
    downsample_dtw,
    topk_per_tgt,
    enable_knn=True,
    enable_dtw=True
):
    sim_id = int(df_sim["simulation"].iloc[0])
    df_sim = compute_features(df_sim)
    agents = sorted(df_sim["id_agent"].unique().tolist())
    A = {a: df_sim[df_sim["id_agent"] == a].reset_index(drop=True) for a in agents}
    XY = {a: A[a][["coordinate_x", "coordinate_y"]].to_numpy() for a in agents}
    T = int(df_sim["time"].max()) + 1

    dist_cache, knn_cache = {}, {}
    for s in agents:
        for t in agents:
            if s == t:
                continue
            dist_cache[(s, t)] = pairwise_mean_distance(XY[s], XY[t])
            if enable_knn:
                knn_cache[(s, t)] = knn_alignment(A[s]["heading"].values, A[t]["heading"].values, k=5)

    true_edges = set((ALPHA_ID, f) for f in FOLLOWERS if f in agents)
    rows = []

    for (w0, w1) in window_indices(T, WINDOW_SIZE, WINDOW_STEP):
        med_dist = {
            (s, t): median_distance_window(XY[s], XY[t], w0, w1)
            for s in agents
            for t in agents
            if s != t
        }

        core_store = {}
        for s in agents:
            for t in agents:
                if s == t:
                    continue
                md_glob = dist_cache[(s, t)]
                if (np.isfinite(dist_threshold) and np.isfinite(md_glob) and md_glob > dist_threshold):
                    core_store[(s, t)] = {"skip": True}
                    continue

                xs, xt = A[s].iloc[w0:w1], A[t].iloc[w0:w1]
                r2_auto, gains, weights, ginfo = krl_components(
                    xs, xt, lags=lag_grid, temp=LAG_TEMP
                )
                if not isinstance(gains, (list, tuple)) or len(gains) == 0:
                    core_store[(s, t)] = {
                        "skip": False,
                        "c_krl": np.nan,
                        "lag_star": lag_grid[0],
                        "gate": 0.0,
                        "r2_auto": np.nan,
                        "r2_gain_max": np.nan,
                        "lag_cons": np.nan
                    }
                    continue

                g = np.nan_to_num(np.asarray(gains, float), nan=0.0)
                w = np.nan_to_num(np.asarray(weights, float), nan=0.0)
                c_krl = float((g * w).sum())
                lag_star = int(lag_grid[int(np.argmax(w))])

                sh, th = xs["heading"].values, xt["heading"].values
                aln = knn_cache.get((s, t), np.nan)
                if np.isnan(aln):
                    try:
                        aln = float(np.nanmean(np.cos(th - sh)))
                    except Exception:
                        aln = np.nan
                md = med_dist[(s, t)]
                gate = compute_gate(
                    dist_med=md if np.isfinite(md) else TAU_DIST * 10.0,
                    align_mean=aln if np.isfinite(aln) else 0.0,
                    r2_gain_max=float(np.nanmax(g))
                    if np.isfinite(np.nanmax(g))
                    else 0.0,
                    lag_consistency=ginfo.get("lag_consistency", 0.0),
                )

                core_store[(s, t)] = {
                    "skip": False,
                    "c_krl": c_krl,
                    "lag_star": lag_star,
                    "gate": gate,
                    "r2_auto": ginfo["r2_auto"],
                    "r2_gain_max": ginfo["r2_gain_max"],
                    "lag_cons": ginfo["lag_consistency"]
                }

        for t in agents:
            for s in agents:
                if s == t:
                    continue
                st = core_store[(s, t)]
                if st.get("skip", False):
                    rows.append(
                        {
                            "simulation": sim_id,
                            "win_start": w0,
                            "win_end": w1,
                            "src": s,
                            "tgt": t,
                            "krlflow": np.nan,
                            "tsmi": np.nan,
                            "te": np.nan,
                            "gc": np.nan,
                            "spatial": dist_cache[(s, t)],
                            "knn": np.nan,
                            "dtw": np.nan,
                            "gate": np.nan,
                            "r2_auto": np.nan,
                            "r2_gain_max": np.nan,
                            "lag_cons": np.nan,
                            "is_true": int((s, t) in true_edges),
                        }
                    )
                    continue

                xs, xt = A[s].iloc[w0:w1], A[t].iloc[w0:w1]
                sh, th = xs["heading"].values, xt["heading"].values
                sp, ac = xs["speed"].values, xs["accel"].values
                lag_star = st["lag_star"]
                gate = st["gate"]
                w_samples = np.full(len(sh), gate, float)

                c_tsmi = tsmi_over_lags(sh,th,lag_grid=lag_grid,weights=w_samples,stride=stride,lag_weights=w)

                Xk = _build_src_kinematics(sh, xs["turn_rate"].values, sp, ac, lag_star)
                Xr = _build_relational(xs, xt, lag_star)
                Z = np.column_stack([Xk, Xr])
                Z = np.nan_to_num(np.asarray(Z, float), nan=0.0)

                Yh = xt["heading"].values
                nbins_te = select_te_nbins_from_signal(Yh, TE_NBINS)

                c_te = np.nan
                if HAS_IDTXL:
                    c_te = te_core_partial_idtxl(
                        sh,
                        th,
                        lag=lag_star,
                        z_feat=Z,
                        nbins=nbins_te,
                        use_z=False,
                        max_z_dims=2
                    )


                gc_maxlag = _choose_gc_maxlag(sh, th, GC_MAXLAGS)
                try:
                    c_gc = gc_core_arx(
                        sh,
                        th,
                        lag=lag_star,
                        z_feat=Z,
                        maxlag=gc_maxlag
                    )
                except Exception:
                    c_gc = np.nan

                c_dtw = np.nan
                if enable_dtw:
                    ds = max(1, downsample_dtw)
                    Axy = XY[s][w0:w1][::ds]
                    Bxy = XY[t][w0:w1][::ds]
                    try:
                        c_dtw = dtw_distance_xy(Axy, Bxy, radius=5)
                    except Exception:
                        c_dtw = np.nan

                rows.append(
                    {
                        "simulation": sim_id,
                        "win_start": w0,
                        "win_end": w1,
                        "src": s,
                        "tgt": t,
                        "krlflow": st["c_krl"],
                        "tsmi": c_tsmi,
                        "te": c_te,
                        "gc": c_gc,
                        "spatial": med_dist[(s, t)],
                        "knn": knn_cache.get((s, t), np.nan),
                        "dtw": c_dtw,
                        "gate": float(gate),
                        "r2_auto": st["r2_auto"],
                        "r2_gain_max": st["r2_gain_max"],
                        "lag_cons": st["lag_cons"],
                        "is_true": int((s, t) in true_edges),
                    }
                )
    return pd.DataFrame(rows)


ABLATIONS = [
    "krlflow_only",
    "krlflow_spatial",
    "krlflow_knn_align",
    "krlflow_dtw",
    "tsmi_only",
    "te_only",
    "gc_only",
    "tsmi_spatial",
    "tsmi_knn_align",
    "tsmi_dtw",
    "te_spatial",
    "te_knn_align",
    "te_dtw",
    "gc_spatial",
    "gc_knn_align",
    "gc_dtw",
    "tsmi_te",
    "tsmi_plus_krl",
    "te_plus_krl",
    "gc_plus_krl",
    "tsmi_te_plus_krl",
]


def fuse_scores(M, ablation, w_tsmi=1.0, w_te=1.0, w_gc=1.0, w_krl=1.0):
    N = len(M)

    def col(c):
        return M[c].values if c in M.columns else np.zeros(N)

    if HAS_SK:
        Z = {k: (_zscore_scipy(col(k)) if k in M.columns else np.zeros(N)) for k in ["tsmi", "te", "gc", "krlflow"]}
    else:

        def _z(v):
            v = np.asarray(v, float)
            mu, sd = np.nanmean(v), np.nanstd(v)
            return (v - mu) / (sd + 1e-12)

        Z = {k: (_z(col(k)) if k in M.columns else np.zeros(N)) for k in ["tsmi", "te", "gc", "krlflow"]}

    gate = np.clip(np.nan_to_num(np.asarray(col("gate"), float), nan=0.0), 0.0, 1.0)

    krl_vec = Z.get("krlflow", np.zeros(N))
    krl_g = (gate**1.5) * np.nan_to_num(np.asarray(krl_vec, float), 0.0)

    def blend(base, krl, u, lam=0.7):
        beta = np.clip(
            lam * gate * (1.0 - np.tanh(np.abs(u))),
            0.0,
            1.0,
        )
        return (1 - beta) * base + beta * krl

    if ablation == "tsmi_only":
        return w_tsmi * np.nan_to_num(np.asarray(Z["tsmi"], float), 0.0)
    if ablation == "te_only":
        return w_te * np.nan_to_num(np.asarray(Z["te"], float), 0.0)
    if ablation == "gc_only":
        return w_gc * np.nan_to_num(np.asarray(Z["gc"], float), 0.0)

    if ablation == "tsmi_plus_krl":
        base = w_tsmi * np.nan_to_num(np.asarray(Z["tsmi"], float), 0.0)
        return blend(base, w_krl * krl_g, u=Z["tsmi"])
    if ablation == "te_plus_krl":
        base = w_te * np.nan_to_num(np.asarray(Z["te"], float), 0.0)
        return blend(base, w_krl * krl_g, u=Z["te"])
    if ablation == "gc_plus_krl":
        base = w_gc * np.nan_to_num(np.asarray(Z["gc"], float), 0.0)
        return blend(base, w_krl * krl_g, u=Z["gc"])
    if ablation == "tsmi_te_plus_krl":
        base = w_tsmi * np.nan_to_num(np.asarray(Z["tsmi"], float), 0.0) + w_te * np.nan_to_num(np.asarray(Z["te"], float), 0.0)
        return blend(base, w_krl * krl_g, u=(Z["tsmi"] + Z["te"]) / 2.0)

    return (
        w_krl * np.nan_to_num(np.asarray(Z.get("krlflow", np.zeros(N)), float), 0.0)
        + w_tsmi * np.nan_to_num(np.asarray(Z["tsmi"], float), 0.0)
        + w_te * np.nan_to_num(np.asarray(Z["te"], float), 0.0)
        + w_gc * np.nan_to_num(np.asarray(Z["gc"], float), 0.0)
    )


WEIGHT_SEEDS = [
    (1.2, 0.4, 0.3, 0.3),
    (1.0, 0.2, 0.2, 0.2),
    (0.8, 0.4, 0.5, 0.5),
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0)
]


def safe_auc(y, s):
    try:
        return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan
    except Exception:
        return np.nan


def safe_aupr(y, s):
    try:
        return float(average_precision_score(y, s))
    except Exception:
        return np.nan


def precision_recall_f1_at_k(y_true, y_score, k):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    idx = np.argsort(-y_score)[: max(1, k)]
    yk = y_true[idx]
    tp = float(np.sum(yk == 1))
    fp = float(len(yk) - tp)
    fn = float(np.sum(y_true) - tp)
    prec = tp / max(1.0, tp + fp)
    rec = tp / max(1.0, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return prec, rec, f1


def hits_at_k(y_true, y_score, k):
    idx = np.argsort(-np.asarray(y_score))[: max(1, k)]
    return float(np.sum(np.asarray(y_true)[idx] == 1))


def ndcg_at_k(y_true, y_score, k):
    idx = np.argsort(-np.asarray(y_score))[: max(1, k)]
    rel = np.asarray(y_true)[idx].astype(float)
    dcg = np.sum((2**rel - 1) / np.log2(np.arange(2, len(rel) + 2)))
    rel_sorted = np.sort(np.asarray(y_true))[::-1][: len(idx)].astype(float)
    idcg = (np.sum((2**rel_sorted - 1) / np.log2(np.arange(2, len(rel_sorted) + 2)))+ 1e-12)
    return float(dcg / idcg)


def adaptive_weight_search(M, ablation):
    y = M["is_true"].values
    best = None
    best_auc = -1.0
    for w in WEIGHT_SEEDS:
        w_krl, w_tsmi, w_te, w_gc = w[0], w[1], w[2], w[3]
        sc = fuse_scores(M, ablation, w_tsmi, w_te, w_gc, w_krl)
        au = safe_auc(y, sc)
        if ((np.isnan(au) and best is None) or (not np.isnan(au) and (au > best_auc or np.isnan(best_auc)))):
            best, best_auc = (w_krl, w_tsmi, w_te, w_gc, 0, 0, 0), au
    if best is None:
        best = (1.0, 1.0, 0.5, 0.5, 0, 0, 0)
    return best, best_auc


def significance_under_gate_edges(M, metrics=("tsmi", "te", "gc"), n_surr=N_SURROGATES):
    rows = []
    if "gate" not in M.columns:
        return pd.DataFrame()

    for (s, t), sub in M.groupby(["src", "tgt"]):
        gates = sub["gate"].values
        if np.all(~np.isfinite(gates)):
            continue
        for mname in metrics:
            if mname not in sub.columns:
                continue
            scores = sub[mname].values
            if np.sum(np.isfinite(scores)) < 10:
                continue
            stat_real = gate_weighted_stat(scores, gates)
            if not np.isfinite(stat_real):
                continue
            surr_stats = []
            B = max(10, n_surr)
            for _ in range(B):
                surr_scores = np.random.permutation(scores)
                surr_stats.append(gate_weighted_stat(surr_scores, gates))
            surr_stats = np.asarray(surr_stats, float)
            m = np.isfinite(surr_stats)
            if m.sum() == 0:
                continue
            p_emp = float((1.0 + np.sum(surr_stats[m] >= stat_real))/ (float(m.sum()) + 1.0))
            delta = cliffs_delta(np.array([stat_real]), surr_stats[m])
            rows.append(
                {
                    "src": int(s),
                    "tgt": int(t),
                    "metric": mname,
                    "stat_real": stat_real,
                    "stat_surr_mean": float(np.nanmean(surr_stats[m])),
                    "p_emp": p_emp,
                    "cliffs_delta": float(delta),
                    "n_surr": int(m.sum())
                }
            )
    return pd.DataFrame(rows)


def evaluate_simulation(df_sim):
    t0 = time.time()
    sim_id = int(df_sim["simulation"].iloc[0])
    log(f"Sim {sim_id}: start")
    df_sim = compute_features(df_sim)

    grid = [
        (s, l, nb, gc, d)
        for s in STRIDES
        for l in [LAGS]
        for nb in [TE_NBINS]
        for gc in [GC_MAXLAGS]
        for d in DIST_THRESHOLDS
    ]
    abl_rows = []
    for gi, (stride, lag_grid, nbins_grid, gc_grid, dthr) in enumerate(grid, 1):
        log(f"  [{gi}/{len(grid)}] stride={stride} lags={lag_grid} nbinsTE={nbins_grid} gcmax={gc_grid} dthr={dthr}")
        M = compute_edge_scores_window_core(
            df_sim,
            lag_grid,
            stride,
            nbins_grid,
            gc_grid,
            dthr,
            downsample_dtw=max(1, 4 * stride),
            topk_per_tgt=TOPK_PER_TGT,
            enable_knn=True,
            enable_dtw=True
        )

        if USE_CALIBRATION and HAS_SK:
            for m in ["tsmi", "te", "gc", "krlflow"]:
                if m in M.columns:
                    y = M["is_true"].values
                    s = M[m].values
                    mask = np.isfinite(s) & np.isfinite(y)
                    if (mask.sum() >= 50 and len(np.unique(y[mask])) >= 2):
                        iso = IsotonicRegression(out_of_bounds="clip")
                        try:
                            iso.fit(s[mask], y[mask])
                            sc = np.full_like(s, np.nan, float)
                            sc[mask] = iso.transform(s[mask])
                            M[m] = sc
                        except Exception:
                            pass

        for abl in ABLATIONS:
            w_star, _ = adaptive_weight_search(M, abl)
            w_krl, w_tsmi, w_te, w_gc = (
                w_star[0],
                w_star[1],
                w_star[2],
                w_star[3]
            )
            sc = fuse_scores(M, abl, w_tsmi, w_te, w_gc, w_krl)
            y = M["is_true"].values
            auc = safe_auc(y, sc)
            aupr = safe_aupr(y, sc)
            d = cliffs_delta(sc[M["is_true"] == 1], sc[M["is_true"] == 0])
            k_eval = min(MAIN_K, max(1, int(0.2 * len(sc))))
            precK, recK, f1K = precision_recall_f1_at_k(y, sc, k_eval)
            hitsK = hits_at_k(y, sc, k_eval)
            ndcgK = ndcg_at_k(y, sc, k_eval)
            abl_rows.append(
                {
                    "simulation": sim_id,
                    "stride": stride,
                    "lag": max(lag_grid),
                    "nbins_te": max(TE_NBINS),
                    "gc_maxlag": max(GC_MAXLAGS),
                    "dist_thr": dthr,
                    "ablation": abl,
                    "w_krl": w_krl,
                    "w_tsmi": w_tsmi,
                    "w_te": w_te,
                    "w_gc": w_gc,
                    "w_spatial": 0,
                    "w_knn": 0,
                    "w_dtw": 0,
                    "auroc": auc,
                    "aupr": aupr,
                    "cliffs_delta": d,
                    "prec_at_k": precK,
                    "recall_at_k": recK,
                    "f1_at_k": f1K,
                    "hits_at_k": hitsK,
                    "ndcg_at_k": ndcgK
                }
            )

        M.to_csv(os.path.join(CSV_DIR,f"sim{sim_id}_edges_stride{stride}_d{int(dthr if np.isfinite(dthr) else 9999)}.csv",),index=False)

    ABL = pd.DataFrame(abl_rows)
    ABL.to_csv(os.path.join(CSV_DIR, f"sim{sim_id}_ablations.csv"),index=False)

    BEST = (ABL.sort_values(["auroc", "aupr"], ascending=[False, False]).iloc[0].to_dict())
    log(f"  Best config sim {sim_id}: {BEST}")

    BEST_PER_ABL = (ABL.sort_values(["auroc", "aupr"], ascending=[False, False]).groupby("ablation").head(1))
    BEST_PER_ABL.to_csv(os.path.join(CSV_DIR, f"sim{sim_id}_best_by_ablation.csv"),index=False,)

    Mbest = compute_edge_scores_window_core(
        df_sim,
        LAGS,
        int(BEST["stride"]),
        TE_NBINS,
        GC_MAXLAGS,
        float(BEST["dist_thr"]),
        downsample_dtw=max(1, 4 * int(BEST["stride"])),
        topk_per_tgt=TOPK_PER_TGT,
        enable_knn=True,
        enable_dtw=True
    )

    for abl in [
        "krlflow_only",
        "tsmi_only",
        "te_only",
        "gc_only",
        "tsmi_plus_krl",
        "te_plus_krl",
        "gc_plus_krl",
        "tsmi_te",
        "tsmi_te_plus_krl",
    ]:
        w_star, _ = adaptive_weight_search(Mbest, abl)
        sc = fuse_scores(Mbest, abl, w_star[1], w_star[2], w_star[3], w_star[0])
        y = Mbest["is_true"].values
        (fpr,tpr,thr_roc,auc,prec,rec,thr_pr,aupr) = roc_pr_curves(y, sc)

        plt.figure()
        plt.plot(rec, prec)
        plt.title(f"Precision-Recall - sim {sim_id} - {abl} (AUPR={aupr:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"sim{sim_id}_PR_{abl}.png"),dpi=300,)
        plt.close()

        plt.figure()
        plt.plot(fpr, tpr)
        plt.title(f"ROC - sim {sim_id} - {abl} (AUC={auc:.3f})")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"sim{sim_id}_ROC_{abl}.png"),dpi=300,)
        plt.close()

    def leader_evolution(Mb, abl, label):
        w_star, _ = adaptive_weight_search(Mb, abl)
        sc = fuse_scores(Mb, abl, w_star[1], w_star[2], w_star[3], w_star[0])
        Mb2 = Mb.copy()
        Mb2["score"] = sc
        rows = []
        for (w0, w1), sub in Mb2.groupby(["win_start", "win_end"]):
            rk = sub.groupby("src")["score"].sum().sort_values(ascending=False)
            rows.append({"win_end": int(w1), "leader": int(rk.index[0])})
        df = pd.DataFrame(rows)
        df["label"] = label
        return df

    EV = pd.concat(
        [
            leader_evolution(Mbest, "tsmi_only", "TSMI"),
            leader_evolution(Mbest, "te_only", "TE"),
            leader_evolution(Mbest, "gc_only", "GC"),
            leader_evolution(Mbest, "krlflow_only", "KRL-Flow"),
            leader_evolution(Mbest, "tsmi_plus_krl", "TSMI+KRL"),
            leader_evolution(Mbest, "te_plus_krl", "TE+KRL"),
            leader_evolution(Mbest, "gc_plus_krl", "GC+KRL")
        ],
        ignore_index=True,
    )
    EV.to_csv(os.path.join(CSV_DIR, f"sim{sim_id}_leader_evolution_compare.csv"),index=False)
    plt.figure()
    for lab, sub in EV.groupby("label"):
        plt.plot(sub["win_end"], sub["leader"], marker="o", label=lab)
    plt.legend()
    plt.title(f"Leader evolution - sim {sim_id}")
    plt.xlabel("time (window end)")
    plt.ylabel("leader agent (top-1)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"sim{sim_id}_leader_evolution_compare.png"),dpi=300)
    plt.close()

    SIG = significance_under_gate_edges(Mbest, metrics=("tsmi", "te", "gc"), n_surr=N_SURROGATES)
    if not SIG.empty:
        SIG.to_csv(os.path.join(CSV_DIR, f"sim{sim_id}_significance_under_gate.csv"),index=False)

    elapsed = time.time() - t0

    with open(os.path.join(OUT_DIR, f"sim{sim_id}_stats.json"), "w") as f:
        json.dump(
            {
                "simulation": sim_id,
                "best_config": BEST,
                "has_sklearn": HAS_SK,
                "has_statsmodels": HAS_STATSMODELS,
                "has_idtxl": HAS_IDTXL,
                "n_surrogates": N_SURROGATES,
                "bootstrap_B": BOOTSTRAP_B,
                "window_size": WINDOW_SIZE,
                "window_step": WINDOW_STEP,
                "topk_per_tgt": TOPK_PER_TGT,
                "lag_temp": LAG_TEMP,
                "ridge_alpha": RIDGE_ALPHA,
                "use_calibration": USE_CALIBRATION,
                "runtime_seconds": float(elapsed),
            },
            f,
            indent=2
        )

    def save_ranking(Mb, abl, title, fname):
        w_star, _ = adaptive_weight_search(Mb, abl)
        sc = fuse_scores(Mb, abl, w_star[1], w_star[2], w_star[3], w_star[0])
        series = (Mb.assign(score=sc)
            .groupby("src")["score"]
            .sum()
            .sort_values(ascending=False)
        )
        series.to_csv(os.path.join( CSV_DIR, f"sim{sim_id}_ranking_{abl}.csv"))
        colors = []

        def col(a):
            if a == ALPHA_ID:
                return "#2ecc71"
            if a in FOLLOWERS:
                return "#3498db"
            return "#e67e22"

        for a in series.index:
            colors.append(col(int(a)))
        plt.figure(figsize=(max(8, len(series) * 0.18), 4.5))
        plt.bar(series.index.astype(str), series.values, color=colors)
        plt.title(title)
        plt.xlabel("Agent (src)")
        plt.ylabel("Aggregated score")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, fname),dpi=300)
        plt.close()
        return series

    reps = [
        "krlflow_only",
        "tsmi_only",
        "te_only",
        "gc_only",
        "tsmi_plus_krl",
        "te_plus_krl",
        "gc_plus_krl",
        "tsmi_te",
        "tsmi_te_plus_krl"
    ]
    for abl in reps:
        save_ranking(Mbest,abl,f"Ranking - {abl} (sim {sim_id})",f"sim{sim_id}_ranking_{abl}.png",)

    return ABL, elapsed


MAIN_METHODS = [
    "tsmi_only",
    "te_only",
    "gc_only",
    "krlflow_only",
    "krlflow_spatial",
    "krlflow_knn_align",
    "krlflow_dtw",
    "tsmi_plus_krl",
    "te_plus_krl",
    "gc_plus_krl",
    "tsmi_te",
    "tsmi_te_plus_krl"
]
PAIR_TESTS = [
    ("tsmi_only", "tsmi_plus_krl"),
    ("te_only", "te_plus_krl"),
    ("gc_only", "gc_plus_krl"),
    ("tsmi_te", "tsmi_te_plus_krl")
]


def mean_ci95(x, B=2000):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    boot = [np.mean(np.random.choice(x, size=len(x), replace=True)) for _ in range(B)]
    m = float(np.mean(boot))
    lo = float(np.percentile(boot, 2.5))
    hi = float(np.percentile(boot, 97.5))
    return (m, lo, hi)


def build_main_table(all_abl):
    best_by = (
        all_abl.sort_values(["auroc", "aupr"], ascending=[False, False])
        .groupby(["simulation", "ablation"])
        .head(1)
    )
    rows = []
    for m in MAIN_METHODS:
        sub = best_by[best_by["ablation"] == m]
        for metric in [
            "auroc",
            "aupr",
            "hits_at_k",
            "ndcg_at_k",
            "f1_at_k"
        ]:
            mean, lo, hi = mean_ci95(sub[metric].values, B=2000)
            rows.append(
                {
                    "method": m,
                    "metric": metric,
                    "mean": mean,
                    "ci95_lo": lo,
                    "ci95_hi": hi
                }
            )
    return pd.DataFrame(rows)


def permutation_pair_tests(all_abl):
    best_by = (
        all_abl.sort_values(["auroc", "aupr"], ascending=[False, False])
        .groupby(["simulation", "ablation"])
        .head(1)
    )
    rows = []
    for A, B in PAIR_TESTS:
        S = sorted(
            set(best_by[best_by["ablation"] == A]["simulation"]).intersection(
                set(
                    best_by[
                        best_by["ablation"] == B
                    ]["simulation"]
                )
            )
        )
        a = []
        b = []
        for s in S:
            a.append(
                float(
                    best_by[
                        (best_by["ablation"] == A)
                        & (best_by["simulation"] == s)
                    ]["auroc"]
                )
            )
            b.append(
                float(
                    best_by[
                        (best_by["ablation"] == B)
                        & (best_by["simulation"] == s)
                    ]["auroc"]
                )
            )
        a, b = np.asarray(a), np.asarray(b)
        if len(a) == 0:
            rows.append(
                {
                    "pair": f"{A} vs {B}",
                    "delta_mean": np.nan,
                    "cliffs_delta": np.nan,
                    "perm_p": np.nan,
                }
            )
            continue
        delta = float(np.nanmean(b - a))
        cd = cliffs_delta(b, a)
        n = 10000
        cnt = 0
        for _ in range(n):
            sign = np.where(
                np.random.rand(len(a)) < 0.5, 1.0, -1.0
            )
            val = np.mean(sign * (b - a))
            if val >= delta:
                cnt += 1
        p = (cnt + 1.0) / (n + 1.0)
        rows.append(
            {
                "pair": f"{A} vs {B}",
                "delta_mean": delta,
                "cliffs_delta": cd,
                "perm_p": p,
            }
        )
    return pd.DataFrame(rows)


def run_hp_sensitivity_single_sim(df_sim, sim_id, methods=None):
    if methods is None:
        methods = [
            "tsmi_plus_krl",
            "te_plus_krl",
            "gc_plus_krl",
            "tsmi_te_plus_krl"
        ]

    global LAG_TEMP, TAU_DIST, TAU_ALIGN, TAU_R2, TAU_LCONS
    rows = []
    base_lag_grid = LAGS

    for cfg in HP_GRID:
        old = (
            LAG_TEMP,
            TAU_DIST,
            TAU_ALIGN,
            TAU_R2,
            TAU_LCONS
        )
        LAG_TEMP = cfg["LAG_TEMP"]
        TAU_DIST = cfg["TAU_DIST"]
        TAU_ALIGN = cfg["TAU_ALIGN"]
        TAU_R2 = cfg["TAU_R2"]
        TAU_LCONS = cfg["TAU_LCONS"]

        log(
            f"[HP] Sim {sim_id} – profile={cfg['name']} "
            f"LAG_TEMP={LAG_TEMP:.3f}, TAU_DIST={TAU_DIST:.2f}, TAU_ALIGN={TAU_ALIGN:.3f}"
        )

        t0 = time.time()
        M = compute_edge_scores_window_core(
            df_sim,
            base_lag_grid,
            stride=1,
            nbins_te_grid=TE_NBINS,
            gc_maxlag_grid=GC_MAXLAGS,
            dist_threshold=np.inf,
            downsample_dtw=4,
            topk_per_tgt=TOPK_PER_TGT,
            enable_knn=True,
            enable_dtw=False
        )
        elapsed = time.time() - t0

        for abl in methods:
            w_star, _ = adaptive_weight_search(M, abl)
            sc = fuse_scores(M, abl, w_star[1], w_star[2], w_star[3], w_star[0])
            y = M["is_true"].values
            auc = safe_auc(y, sc)
            aupr = safe_aupr(y, sc)
            k_eval = min(MAIN_K, max(1, int(0.2 * len(sc))))
            precK, recK, f1K = precision_recall_f1_at_k(y, sc, k_eval)
            hitsK = hits_at_k(y, sc, k_eval)
            ndcgK = ndcg_at_k(y, sc, k_eval)
            rows.append(
                {
                    "hp_profile": cfg["name"],
                    "simulation": int(sim_id),
                    "ablation": abl,
                    "LAG_TEMP": cfg["LAG_TEMP"],
                    "TAU_DIST": cfg["TAU_DIST"],
                    "TAU_ALIGN": cfg["TAU_ALIGN"],
                    "TAU_R2": cfg["TAU_R2"],
                    "TAU_LCONS": cfg["TAU_LCONS"],
                    "runtime_seconds": float(elapsed),
                    "auroc": auc,
                    "aupr": aupr,
                    "prec_at_k": precK,
                    "recall_at_k": recK,
                    "f1_at_k": f1K,
                    "hits_at_k": hitsK,
                    "ndcg_at_k": ndcgK,
                }
            )

        LAG_TEMP, TAU_DIST, TAU_ALIGN, TAU_R2, TAU_LCONS = old

    return pd.DataFrame(rows)

def main():
    log("Load data...")
    DF = load_all_simulations(DATA_DIR, GLOB_PATTERN)
    sims = sorted(DF["simulation"].unique().tolist())
    ALL = []
    runtime_rows = []

    for sid in sims:
        ABL, elapsed = evaluate_simulation(DF[DF["simulation"] == sid].copy())
        ALL.append(ABL)
        runtime_rows.append(
            {
                "simulation": int(sid),
                "runtime_seconds": float(elapsed),
            }
        )

    ALL = pd.concat(ALL, ignore_index=True)
    ALL.to_csv(
        os.path.join(CSV_DIR, "ALL_sim_ablations_raw.csv"),
        index=False
    )

    log("Main table...")
    MAIN = build_main_table(ALL)
    MAIN.to_csv(
        os.path.join(CSV_DIR, "main_table_mean_ci95.csv"),
        index=False
    )

    log("Pairwise tests...")
    TESTS = permutation_pair_tests(ALL)
    TESTS.to_csv(
        os.path.join(CSV_DIR, "baseline_vs_plusKRL_tests.csv"),
        index=False
    )

    RUNTIME = pd.DataFrame(runtime_rows)
    RUNTIME.to_csv(
        os.path.join(OUT_DIR, "runtime_per_sim.csv"),
        index=False
    )

    hp_summary = None
    if HP_SWEEP_FLAG and len(sims) > 0:
        log(f"Running hyperparameter sensitivity sweep on sim {sims[0]}...")
        hp_summary = run_hp_sensitivity_single_sim(DF[DF["simulation"] == sims[0]].copy(), sims[0])
        hp_summary.to_csv(
            os.path.join(
                CSV_DIR, "hp_sensitivity_single_sim.csv"
            ),
            index=False
        )

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(
            {
                "n_sims": int(len(sims)),
                "has_idtxl": HAS_IDTXL,
                "total_runtime_seconds": float(
                    RUNTIME["runtime_seconds"].sum()
                ),
                "hp_sweep_enabled": bool(HP_SWEEP_FLAG),
                "hp_profiles": [cfg["name"] for cfg in HP_GRID]
                if HP_SWEEP_FLAG
                else [],
            },
            f,
            indent=2,
        )

def _idtxl_set_discrete_ts(data_obj, V, normalise=False):
    V_arr = np.asarray(V).astype(int)

    if V_arr.ndim == 2:
        V_arr = V_arr[:, :, np.newaxis]

    if hasattr(data_obj, "submit_discrete_time_series"):
        try:
            return data_obj.submit_discrete_time_series(V_arr, normalise=normalise)
        except TypeError:
            return data_obj.submit_discrete_time_series(V_arr)

    if hasattr(data_obj, "set_discrete_time_series"):
        try:
            return data_obj.set_discrete_time_series(V_arr, normalise=normalise)
        except TypeError:
            return data_obj.set_discrete_time_series(V_arr)

    if hasattr(data_obj, "set_data"):
        try:
            return data_obj.set_data(V_arr, dim_order="psr")
        except TypeError:
            return data_obj.set_data(V_arr)

    raise AttributeError(
        f"IDTxl Data object has no known method to set discrete time series. "
        f"Available methods: {dir(data_obj)}"
    )



if __name__ == "__main__":
    main()

