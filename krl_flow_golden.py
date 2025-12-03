import os, json, math, time
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HAS_SK = False
HAS_STATSMODELS = False
HAS_IDTXL = False
HAS_SCIPY = False
USE_IDTXL = os.environ.get("KRL_USE_IDTXL", "1") == "1"

try:
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,roc_curve
    from scipy.stats import zscore as _zscore_scipy, mannwhitneyu, ttest_ind
    HAS_SK = True
    HAS_SCIPY = True
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

CSV_IN = os.environ.get("KRL_GSHINER_CSV", "files/golden/entropy-3134626-supplementary.csv")
OUT_DIR = os.environ.get("KRL_OUT_DIR", "results_krl_out_gshiner")
FIG_DIR = os.path.join(OUT_DIR, "figs"); os.makedirs(FIG_DIR, exist_ok=True)
CSV_DIR = os.path.join(OUT_DIR, "csv");  os.makedirs(CSV_DIR, exist_ok=True)
LABELS_CSV = "files/golden/golden_labels.csv"

def _sanitize_filename(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(name))

def save_csv_for_figure(fig_basename: str, df: pd.DataFrame) -> None:
    if df is None:
        return
    try:
        out = os.path.join(CSV_DIR, f"{_sanitize_filename(fig_basename)}.csv")
        df.to_csv(out, index=False)
    except Exception as e:
        try:
            log(f"[WARN] Could not save CSV for figure '{fig_basename}': {e}")
        except Exception:
            pass

def _tuple_from_env(name, default):
    v = os.environ.get(name, default)
    parts = [x.strip() for x in v.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{name} deve ter 2 colunas separadas por vírgula. Recebido: {v}")
    return parts[0], parts[1]

THETA_COLS = _tuple_from_env("KRL_POS_THETA_COLS", "theta0,theta1")
RADIUS_COLS = _tuple_from_env("KRL_POS_RADIUS_COLS", "r0,r1")
HEADING_COLS= _tuple_from_env("KRL_HEADING_COLS", "psi0,psi1")
ALIGN_COLS = _tuple_from_env("KRL_ALIGN_COLS", "align_01_angle,align_10_angle")


WINDOW_SIZE = int(os.environ.get("KRL_WINDOW_SIZE", "600"))
WINDOW_STEP = int(os.environ.get("KRL_WINDOW_STEP",  "600"))
LAGS = [1, 2, 3, 4]
STRIDES = [1, 2, 3]
TE_NBINS = [6, 8, 12]
GC_MAXLAGS = [2, 4, 6]

LAG_TEMP = float(os.environ.get("KRL_LAG_TEMP", "0.5"))
RIDGE_ALPHA = float(os.environ.get("KRL_RIDGE_ALPHA", "1.0"))
MAIN_K = int(os.environ.get("KRL_MAIN_K", "2"))

# Gates
TAU_DIST = float(os.environ.get("KRL_TAU_DIST", "0.20"))
TAU_ALIGN = float(os.environ.get("KRL_TAU_ALIGN", "0.20"))
TAU_R2 = float(os.environ.get("KRL_TAU_R2", "0.01"))
TAU_LCONS = float(os.environ.get("KRL_TAU_LCONS", "0.4"))
TAU_PSI_VAR = float(os.environ.get("KRL_TAU_PSI_VAR", "0.15"))

USE_CALIBRATION = os.environ.get("KRL_USE_CALIB", "1") == "1"
RANDOM_SEED = 7
np.random.seed(RANDOM_SEED)


TIME_STATS = defaultdict(float)

def log(msg: str):
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

def window_indices(T, win, step):
    start = 0
    while start < T:
        end = min(start + win, T)
        yield start, end
        if end == T:
            break
        start += step

def precision_recall_f1_at_k(y_true, y_score, k):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    idx = np.argsort(-y_score)[:max(1, k)]
    yk = y_true[idx]
    tp = float(np.sum(yk == 1))
    fp = float(len(yk) - tp)
    fn = float(np.sum(y_true) - tp)
    prec = tp / max(1.0, tp + fp)
    rec = tp / max(1.0, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return prec, rec, f1

def hits_at_k(y_true, y_score, k):
    idx = np.argsort(-np.asarray(y_score))[:max(1, k)]
    return float(np.sum(np.asarray(y_true)[idx] == 1))

def ndcg_at_k(y_true, y_score, k):
    idx = np.argsort(-np.asarray(y_score))[:max(1, k)]
    rel = np.asarray(y_true)[idx].astype(float)
    dcg = np.sum((2**rel - 1) / np.log2(np.arange(2, len(rel) + 2)))
    rel_sorted = np.sort(np.asarray(y_true))[::-1][:len(idx)].astype(float)
    idcg = np.sum((2**rel_sorted - 1) / np.log2(np.arange(2, len(rel_sorted) + 2))) + 1e-12
    return float(dcg / idcg)

def roc_pr_curves(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    m = np.isfinite(s) & np.isfinite(y)
    if m.sum() < 5 or len(np.unique(y[m])) < 2:
        return (
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.5]),
            np.nan,
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.nan
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

def circular_corr(x, y, lag):
    x = np.asarray(x)
    y = np.asarray(y)
    xs = pd.Series(x).shift(lag).values
    mask = ~np.isnan(xs) & ~np.isnan(y)
    if mask.sum() < 30:
        return np.nan
    X = np.column_stack([np.sin(xs[mask]), np.cos(xs[mask])])
    Y = np.column_stack([np.sin(y[mask]), np.cos(y[mask])])
    try:
        c = np.corrcoef(X.T, Y.T)[0:2, 2:4]
        return float(np.linalg.norm(c))
    except Exception:
        return np.nan

def cliffs_delta(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    n1, n2 = len(x), len(y)
    greater = 0
    lesser = 0
    for a in x:
        greater += np.sum(a > y)
        lesser += np.sum(a < y)
    return float((greater - lesser) / (n1 * n2 + 1e-12))

def load_gshiner_pair(csv_path: str):
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}

    need = ["t",
        "frame",
        THETA_COLS[0],
        THETA_COLS[1],
        RADIUS_COLS[0],
        RADIUS_COLS[1],
        HEADING_COLS[0],
        HEADING_COLS[1],
        ALIGN_COLS[0],
        ALIGN_COLS[1],
    ]
    for c in need:
        if c not in df.columns:
            if c.lower() in cols:
                df[c] = df[cols[c.lower()]]
            else:
                raise ValueError(f"Coluna obrigatória ausente: {c}")

    def pol_to_xy(r, th):
        return r * np.cos(th), r * np.sin(th)

    x0, y0 = pol_to_xy(df[RADIUS_COLS[0]].values, df[THETA_COLS[0]].values)
    x1, y1 = pol_to_xy(df[RADIUS_COLS[1]].values, df[THETA_COLS[1]].values)

    # Heading (psi)
    h0 = df[HEADING_COLS[0]].values
    h1 = df[HEADING_COLS[1]].values

    # Alignments provided: 0->1 and 1->0 (angles). Converted to cos (degree of alignment)
    al01 = df[ALIGN_COLS[0]].values
    al10 = df[ALIGN_COLS[1]].values
    ca01 = np.cos(al01)
    ca10 = np.cos(al10)

    # approximate speed
    def speed_from_xy(x, y):
        dx = np.r_[np.nan, np.diff(x)]
        dy = np.r_[np.nan, np.diff(y)]
        return np.hypot(dx, dy)

    v0 = speed_from_xy(x0, y0)
    v1 = speed_from_xy(x1, y1)

    t = df["t"].values if "t" in df.columns else df["frame"].values
    sim = 1
    D0 = pd.DataFrame(
        {
            "simulacao": sim,
            "tempo": np.arange(len(t), dtype=int),
            "id_agente": 0,
            "coordenada_x": x0,
            "coordenada_y": y0,
            "velocidade": v0,
            "heading": h0,
            #0 -> 1
            "align_src_to_tgt": ca01, 
            "label": "unknown",
        }
    )
    D1 = pd.DataFrame(
        {
            "simulacao": sim,
            "tempo": np.arange(len(t), dtype=int),
            "id_agente": 1,
            "coordenada_x": x1,
            "coordenada_y": y1,
            "velocidade": v1,
            "heading": h1,
            #1 -> 0
            "align_src_to_tgt": ca10,
            "label": "unknown",
        }
    )
    D = pd.concat([D0, D1], ignore_index=True)
    return D

def _build_auto_target_matrix(tgt_head):
    y_t = np.asarray(tgt_head, float)
    y_tm1 = np.roll(y_t, 1)
    y_tm1[0] = np.nan
    X_auto = np.column_stack([np.sin(y_tm1), np.cos(y_tm1)])
    return X_auto, y_t

def _fit_ridge(X, y, alpha=RIDGE_ALPHA):
    try:
        m = Ridge(alpha=alpha, fit_intercept=True)
        m.fit(X, y)
        return m, m.predict(X)
    except Exception:
        return None, None

def zshift(a, lag):
    return pd.Series(a).shift(lag).values

def _build_src_kinematics(src_head, src_turn, src_speed, src_accel, lag):
    sh = zshift(src_head, lag)
    tr = zshift(src_turn, lag)
    sp = zshift(src_speed, lag)
    ac = zshift(src_accel, lag)
    Xk = np.column_stack([np.sin(sh), np.cos(sh), np.sin(tr), np.cos(tr), zscore(sp), zscore(ac)])
    return Xk

def _build_relational(xs, xt, lag):
    sh = zshift(xs["heading"].values, lag)
    sp = zshift(xs["velocidade"].values, lag)
    sx = zshift(xs["coordenada_x"].values, lag)
    sy = zshift(xs["coordenada_y"].values, lag)

    th = xt["heading"].values
    tx = xt["coordenada_x"].values
    ty = xt["coordenada_y"].values
    tv = xt["velocidade"].values
    dx, dy = tx - sx, ty - sy
    dist = np.sqrt(dx**2 + dy**2)
    bearing = np.arctan2(dy, dx)
    al = zshift(xs["align_src_to_tgt"].values, lag)
    dv = tv - sp
    ratio = np.divide(tv, np.where(np.abs(sp) < 1e-6, np.nan, sp))
    Xr = np.column_stack(
        [
            zscore(dist),
            np.sin(bearing),
            np.cos(bearing),
            np.nan_to_num(al, nan=0.0),
            zscore(dv),
            zscore(ratio)
        ]
    )
    return Xr

def krl_components(xs, xt, lags, temp=LAG_TEMP):
    if not HAS_SK:
        return np.nan, [], [], {}

    tgt_head = xt["heading"].values
    src_head = xs["heading"].values
    turn_src = np.mod(np.diff(src_head, prepend=src_head[0]) + np.pi, 2 * np.pi) - np.pi
    src_speed = xs["velocidade"].values
    src_acc = np.r_[np.nan, np.diff(src_speed)]

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

    # Lag bias due to circular correlation of psi
    cc = {lag: circular_corr(src_head, tgt_head, lag) for lag in lags}
    cc_norm = np.array([cc.get(l, np.nan) for l in lags], float)
    if np.all(np.isnan(cc_norm)):
        cc_norm = np.zeros(len(lags))
    cc_norm = np.nan_to_num(cc_norm, nan=0.0)
    if cc_norm.max() > 0:
        cc_norm = cc_norm / (cc_norm.max() + 1e-12)

    gains = []
    raw_r2 = []
    for i, lag in enumerate(lags):
        Xk = _build_src_kinematics(src_head, turn_src, src_speed, src_acc, lag)
        Xr = _build_relational(xs, xt, lag)
        X = np.column_stack([Xa, Xk, Xr]).astype(float)
        m = ~np.isnan(y) & ~np.isnan(X).any(1)
        if m.sum() < 30:
            gains.append(np.nan)
            raw_r2.append(np.nan)
            continue
        _, yhat = _fit_ridge(X[m], y[m])
        if yhat is None:
            gains.append(np.nan)
            raw_r2.append(np.nan)
            continue
        ss_res = float(np.sum((y[m] - yhat) ** 2))
        r2_full = max(0.0, 1.0 - ss_res / ss_tot)
        raw_r2.append(r2_full)
        gains.append(max(0.0, r2_full - r2_auto) + 0.15 * cc_norm[i])

    g = np.nan_to_num(np.array(gains, float), nan=0.0)
    if np.all(g == 0):
        w = np.zeros_like(g)
    else:
        w = np.exp(g / max(1e-6, temp))
        w /= w.sum() + 1e-12
    lag_cons = float(np.sort(w)[-2:].sum()) if w.size > 1 else float(w.max() if w.size else np.nan)
    return r2_auto, list(g), list(w), {
        "r2_auto": r2_auto,
        "r2_gain_max": float(np.nanmax(g) if len(g) else np.nan),
        "lag_consistency": lag_cons,
        "r2_full_by_lag": raw_r2,
        "ccorr_by_lag": [cc.get(l, np.nan) for l in lags],
    }

def compute_gate(dist_med, align_mean_sym, align_delta, r2_gain_max, lag_consistency, psi_var_tgt):
    x = np.array(
        [
            (TAU_DIST - dist_med) / max(1.0, TAU_DIST),
            (align_mean_sym - TAU_ALIGN) / max(1e-6, abs(TAU_ALIGN)),
            align_delta,
            (r2_gain_max - TAU_R2) / max(1e-6, abs(TAU_R2)),
            (lag_consistency - TAU_LCONS) / max(1e-6, abs(TAU_LCONS)),
            (TAU_PSI_VAR - psi_var_tgt) / max(1e-6, TAU_PSI_VAR),
        ],
        float
    )
    w = np.array([0.6, 1.2, 0.8, 1.0, 1.0, 0.6])
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

def _digitize_multi(M, nbins=8, weights=None):
    qs = np.linspace(0, 1, nbins + 1)
    B = []
    keep = np.ones(len(M), dtype=bool)
    for j in range(M.shape[1]):
        v = M[:, j]
        finite = np.isfinite(v)
        if finite.sum() == 0:
            keep &= np.zeros(len(M), bool)
            continue
        edges = np.unique(np.quantile(v[finite], qs))
        if len(edges) < 3:
            keep &= np.zeros(len(M), bool)
            continue
        b = np.digitize(v, edges[1:-1], right=True)
        b[~finite] = -1
        keep &= b >= 0
        B.append(b)
    if keep.sum() < 50:
        return None, None
    B = [b[keep] for b in B]
    w = None if weights is None else np.nan_to_num(np.asarray(weights, float)[keep], nan=0.0)
    return B, w

def te_core_partial_idtxl(src_sig, tgt_sig, lag, z_feat, nbins=8):
    if not HAS_IDTXL:
        return np.nan

    y_t = pd.Series(tgt_sig).values
    y_tm1 = pd.Series(tgt_sig).shift(1).values
    x_l = pd.Series(src_sig).shift(lag).values

    z_feat = np.asarray(z_feat, float)
    if z_feat.ndim == 1:
        z_feat = z_feat.reshape(-1, 1)

    if z_feat.size > 0:
        M = np.column_stack([y_t, y_tm1, x_l, z_feat])
    else:
        M = np.column_stack([y_t, y_tm1, x_l])

    mask = ~np.isnan(M).any(axis=1)
    M = M[mask]

    if M.shape[0] < 200:
        return np.nan

    Y = M[:, 0:1]
    Xl = M[:, 2:3]

    if M.shape[1] > 3:
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
            "kraskov_k": 4,
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

ABLATIONS = [
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
        beta = np.clip(lam * (gate) * (1.0 - np.tanh(np.abs(u))), 0.0, 1.0)
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
        base = (w_tsmi * np.nan_to_num(np.asarray(Z["tsmi"], float), 0.0) + w_te * np.nan_to_num(np.asarray(Z["te"], float), 0.0))
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
        y = np.asarray(y)
        m = np.isfinite(y) & np.isfinite(s)
        if m.sum() < 5 or len(np.unique(y[m])) < 2:
            return np.nan
        return float(roc_auc_score(y[m], s[m]))
    except Exception:
        return np.nan

def safe_aupr(y, s):
    try:
        y = np.asarray(y)
        m = np.isfinite(y) & np.isfinite(s)
        if m.sum() < 5 or len(np.unique(y[m])) < 2:
            return np.nan
        return float(average_precision_score(y[m], s[m]))
    except Exception:
        return np.nan

def adaptive_weight_search(M, ablation):
    y = M["is_true"].values if "is_true" in M.columns else np.full(len(M), np.nan)
    best = None
    best_auc = -1.0
    for w in WEIGHT_SEEDS:
        w_krl, w_tsmi, w_te, w_gc = w[0], w[1], w[2], w[3]
        sc = fuse_scores(M, ablation, w_tsmi, w_te, w_gc, w_krl)
        au = safe_auc(y, sc)
        if (np.isnan(au) and best is None) or (not np.isnan(au) and (au > best_auc or np.isnan(best_auc))):
            best, best_auc = (w_krl, w_tsmi, w_te, w_gc, 0, 0, 0), au
    if best is None:
        best = (1.0, 1.0, 0.5, 0.5, 0, 0, 0)
    return best, best_auc

def compute_edges_gshiner(D):
    log("Computing edges (KRL core enabled)...")
    sim_id = 1
    agents = [0, 1]
    A = {a: D[D["id_agente"] == a].reset_index(drop=True) for a in agents}
    T = int(D["tempo"].max()) + 1

    rows = []
    for (w0, w1) in window_indices(T, WINDOW_SIZE, WINDOW_STEP):
        def med_dist(a, b):
            ax, ay = (
                A[a]["coordenada_x"].values[w0:w1],
                A[a]["coordenada_y"].values[w0:w1],
            )
            bx, by = (
                A[b]["coordenada_x"].values[w0:w1],
                A[b]["coordenada_y"].values[w0:w1],
            )
            n = min(len(ax), len(bx))
            if n == 0:
                return np.nan
            return float(np.median(np.hypot(ax[:n] - bx[:n], ay[:n] - by[:n])))

        align01 = np.nanmean(A[0]["align_src_to_tgt"].values[w0:w1])
        align10 = np.nanmean(A[1]["align_src_to_tgt"].values[w0:w1])
        align_mean_sym = np.nanmean([align01, align10])
        align_delta = float((align10 - align01))

        psi_var = {
            0: np.nanvar(A[0]["heading"].values[w0:w1]),
            1: np.nanvar(A[1]["heading"].values[w0:w1]),
        }

        for s in agents:
            for t in agents:
                if s == t:
                    continue
                xs, xt = A[s].iloc[w0:w1], A[t].iloc[w0:w1]

                t_k0 = time.time()
                r2_auto, gains, weights, ginfo = krl_components(xs, xt, lags=LAGS, temp=LAG_TEMP)
                TIME_STATS["krl_components"] += time.time() - t_k0

                if not isinstance(gains, (list, tuple)) or len(gains) == 0:
                    c_krl = np.nan
                    lag_star = LAGS[0]
                    gate = 0.0
                    r2_gain_max = np.nan
                    lag_cons = np.nan
                else:
                    g = np.nan_to_num(np.asarray(gains, float), nan=0.0)
                    w = np.nan_to_num(np.asarray(weights, float), nan=0.0)
                    c_krl = float((g * w).sum())
                    lag_star = int(LAGS[int(np.argmax(w))])
                    r2_gain_max = float(np.nanmax(g))
                    lag_cons = ginfo.get("lag_consistency", 0.0)

                    t_gate0 = time.time()
                    gate = compute_gate(
                        dist_med=med_dist(s, t),
                        align_mean_sym=align_mean_sym,
                        align_delta=align_delta
                        if s == 1 and t == 0
                        else (-align_delta),
                        r2_gain_max=r2_gain_max,
                        lag_consistency=lag_cons,
                        psi_var_tgt=psi_var[t],
                    )
                    TIME_STATS["gate"] += time.time() - t_gate0

                sh, th = xs["heading"].values, xt["heading"].values
                sp = xs["velocidade"].values
                ac = np.r_[np.nan, np.diff(xs["velocidade"].values)]
                w_samples = np.full(len(sh), gate, float)

                # TSMI
                t_tsmi0 = time.time()
                c_tsmi = tsmi_core(sh, th, lag=lag_star, weights=w_samples, stride=1)
                TIME_STATS["tsmi"] += time.time() - t_tsmi0

                # features Z
                Xk = _build_src_kinematics(sh, np.r_[np.nan, np.diff(sh)], sp, ac, lag_star)
                Xr = _build_relational(xs, xt, lag_star)
                Z = np.column_stack([Xk, Xr])
                Z = np.nan_to_num(np.asarray(Z, float), nan=0.0)

                # nbins TE
                Yh = xt["heading"].values
                nbins_te = select_te_nbins_from_signal(Yh, TE_NBINS)

                # TE
                t_te0 = time.time()
                if HAS_IDTXL and USE_IDTXL:
                    c_te = te_core_partial_idtxl(
                        xs["heading"].values,
                        xt["heading"].values,
                        lag=lag_star,
                        z_feat=Z,
                        nbins=nbins_te,
                    )
                else:
                    c_te = te_core_partial_idtxl(
                    sh,
                    th,
                    lag=lag_star,
                    z_feat=Z,
                    nbins=nbins_te,
                )

                TIME_STATS["te"] += time.time() - t_te0

                # GC
                t_gc0 = time.time()
                try:
                    c_gc = gc_core_arx(
                        sh,
                        th,
                        lag=lag_star,
                        z_feat=Z,
                        maxlag=GC_MAXLAGS[-1],
                        weights=w_samples,
                    )
                except Exception:
                    c_gc = np.nan
                TIME_STATS["gc"] += time.time() - t_gc0

                rows.append(
                    {
                        "simulacao": sim_id,
                        "win_start": w0,
                        "win_end": w1,
                        "src": s,
                        "tgt": t,
                        "krlflow": c_krl,
                        "tsmi": c_tsmi,
                        "te": c_te,
                        "gc": c_gc,
                        "gate": float(gate),
                        "r2_auto": ginfo.get("r2_auto", np.nan),
                        "r2_gain_max": r2_gain_max
                        if "r2_gain_max" in locals()
                        else np.nan,
                        "lag_cons": lag_cons if "lag_cons" in locals() else np.nan,
                        "is_true": np.nan,
                    }
                )
    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(CSV_DIR, "sim1_edges.csv"), index=False)
    return M

def attach_labels_if_available(M: pd.DataFrame) -> pd.DataFrame:
    if not LABELS_CSV or not os.path.exists(LABELS_CSV):
        log("No ground truth provided (KRL_LABELS_CSV empty). Skipping supervised metrics.")
        return M
    L = pd.read_csv(LABELS_CSV)
    need = {"win_start", "win_end", "leader_src"}
    if not need.issubset(set([c.lower() for c in L.columns])):
        log("LABELS_CSV missing expected columns. Ignoring ground truth.")
        return M
    L.columns = [c.lower() for c in L.columns]
    M2 = M.copy()
    is_true = []
    for _, r in M2.iterrows():
        sub = L[(L["win_start"] == r["win_start"]) & (L["win_end"] == r["win_end"])]
        if len(sub) == 0:
            is_true.append(np.nan)
            continue
        leader = int(sub["leader_src"].iloc[0])
        is_true.append(int(r["src"] == leader))
    M2["is_true"] = is_true
    return M2

def gate_significance_analysis(M: pd.DataFrame, sim_id: int = 1):
    if "gate" not in M.columns:
        log("No 'gate' column found. Skipping gate significance analysis.")
        return None

    gate = np.asarray(M["gate"], float)
    if np.all(~np.isfinite(gate)):
        log("Gate column has no finite values. Skipping gate significance analysis.")
        return None

    q25 = np.nanpercentile(gate, 25.0)
    q75 = np.nanpercentile(gate, 75.0)
    hi_mask = gate >= q75
    lo_mask = gate <= q25

    metrics = ["krlflow", "tsmi", "te", "gc"]
    rows = []
    for m in metrics:
        if m not in M.columns:
            continue
        s = np.asarray(M[m], float)
        hi = s[hi_mask & np.isfinite(s)]
        lo = s[lo_mask & np.isfinite(s)]
        if len(hi) < 10 or len(lo) < 10:
            continue
        mean_hi = float(np.nanmean(hi))
        mean_lo = float(np.nanmean(lo))
        diff = mean_hi - mean_lo
        delta = cliffs_delta(hi, lo)

        p_mw = np.nan
        p_t = np.nan
        if HAS_SCIPY:
            try:
                p_mw = float(mannwhitneyu(hi, lo, alternative="two-sided").pvalue)
            except Exception:
                pass
            try:
                p_t = float(ttest_ind(hi, lo, equal_var=False, nan_policy="omit").pvalue)
            except Exception:
                pass

        rows.append(
            {
                "metric": m,
                "high_gate_q": float(q75),
                "low_gate_q": float(q25),
                "n_high": int(len(hi)),
                "n_low": int(len(lo)),
                "mean_high": mean_hi,
                "mean_low": mean_lo,
                "mean_diff_high_minus_low": diff,
                "cliffs_delta": delta,
                "p_mannwhitney": p_mw,
                "p_ttest": p_t
            }
        )

    if not rows:
        log("Not enough data for gate significance analysis.")
        return None

    df = pd.DataFrame(rows)
    out_path = os.path.join(CSV_DIR, f"sim{sim_id}_gate_significance.csv")
    df.to_csv(out_path, index=False)
    log(f"Gate significance analysis written to {out_path}")
    return df

def summarize_ablation_metrics(M: pd.DataFrame, sim_id: int = 1):
    if "is_true" not in M.columns or not np.isfinite(M["is_true"]).any():
        log("No ground truth available. Skipping ablation metric table.")
        return None

    rows = []
    for abl in ABLATIONS:
        w_star, _ = adaptive_weight_search(M, abl)
        sc = fuse_scores(M, abl, w_star[1], w_star[2], w_star[3], w_star[0])
        y = M["is_true"].values
        auc_edge = safe_auc(y, sc)
        aupr_edge = safe_aupr(y, sc)

        Mb = M.assign(score=sc)
        corr = []
        for (_, _), sub in Mb.groupby(["win_start", "win_end"]):
            agg = sub.groupby("src")["score"].sum()
            if len(agg) == 0:
                continue
            pred_src = int(agg.idxmax())
            gt = sub[sub["is_true"] == 1]
            if len(gt) == 0:
                continue
            true_src = int(gt["src"].iloc[0])
            corr.append(int(pred_src == true_src))
        top1_win_acc = float(np.mean(corr)) if corr else np.nan

        rows.append(
            {
                "ablation": abl,
                "auc_edge": auc_edge,
                "aupr_edge": aupr_edge,
                "top1_window_acc": top1_win_acc
            }
        )

    if not rows:
        log("Ablation metric table ended up empty; nothing to write.")
        return None

    df = pd.DataFrame(rows)
    out_path = os.path.join(CSV_DIR, f"sim{sim_id}_ablation_metrics.csv")
    df.to_csv(out_path, index=False)
    log(f"Ablation metrics written to {out_path}")
    return df


COLOR_0 = "#e67e22"  # orange (Fish 0)
COLOR_1 = "#2ecc71"  # green  (Fish 1)

def evaluate_and_write(M: pd.DataFrame, phase_times=None):
    sim_id = 1
    if phase_times is None:
        phase_times = {}
    t_eval0 = time.time()

    if USE_CALIBRATION and HAS_SK and "is_true" in M.columns and np.isfinite(M["is_true"]).any():
        for m in ["tsmi", "te", "gc", "krlflow"]:
            if m in M.columns:
                y = M["is_true"].values
                s = M[m].values
                mask = np.isfinite(s) & np.isfinite(y)
                if mask.sum() >= 50 and len(np.unique(y[mask])) >= 2:
                    iso = IsotonicRegression(out_of_bounds="clip")
                    try:
                        iso.fit(s[mask], y[mask])
                        sc = np.full_like(s, np.nan, float)
                        sc[mask] = iso.transform(s[mask])
                        M[m] = sc
                    except Exception:
                        pass

    # ranking per ablation
    def save_ranking(Mb, abl, title, fname):
        w_star, _ = adaptive_weight_search(Mb, abl)
        sc = fuse_scores(Mb, abl, w_star[1], w_star[2], w_star[3], w_star[0])
        series = Mb.assign(score=sc).groupby("src")["score"].sum()
        series = series.reindex([0, 1])
        series.to_csv(os.path.join(CSV_DIR, f"sim{sim_id}_ranking_{abl}.csv"))
        colors = [COLOR_0, COLOR_1]
        labels = ["Fish 0", "Fish 1"]
        plt.figure(figsize=(6, 4.5))
        plt.bar([0, 1], series.values, color=colors, tick_label=labels)
        plt.title(title)
        plt.xlabel("Agent (src)")
        plt.ylabel("Aggregated score")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, fname), dpi=130)
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
        save_ranking(M, abl, f"Ranking - {abl} (sim {sim_id})", f"sim{sim_id}_ranking_{abl}.png")

    # leader evolution
    def leader_evolution(Mb, abl, label):
        w_star, _ = adaptive_weight_search(Mb, abl)
        sc = fuse_scores(Mb, abl, w_star[1], w_star[2], w_star[3], w_star[0])
        Mb2 = Mb.copy()
        Mb2["score"] = sc
        rows = []
        for (w0, w1), sub in Mb2.groupby(["win_start", "win_end"]):
            rk = sub.groupby("src")["score"].sum()
            leader = 0 if rk.get(0, -np.inf) >= rk.get(1, -np.inf) else 1
            rows.append({"win_end": int(w1), "leader": int(leader)})
        df = pd.DataFrame(rows)
        df["label"] = label
        return df

    EV = pd.concat(
        [
            leader_evolution(M, "tsmi_only", "TSMI"),
            leader_evolution(M, "te_only", "TE"),
            leader_evolution(M, "gc_only", "GC"),
            leader_evolution(M, "krlflow_only", "KRL-Flow"),
            leader_evolution(M, "tsmi_plus_krl", "TSMI+KRL"),
            leader_evolution(M, "te_plus_krl", "TE+KRL"),
            leader_evolution(M, "gc_plus_krl", "GC+KRL")
        ],
        ignore_index=True,
    )
    EV.to_csv(
        os.path.join(CSV_DIR, f"sim{sim_id}_leader_evolution_compare.csv"), index=False
    )
    plt.figure()
    for lab, sub in EV.groupby("label"):
        plt.plot(sub["win_end"], sub["leader"], marker="o", label=lab)
    plt.yticks([0, 1], ["Fish 0", "Fish 1"])
    plt.legend()
    plt.title(f"Leader evolution – sim {sim_id}")
    plt.xlabel("time (window end)")
    plt.ylabel("leader agent (top-1)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"sim{sim_id}_leader_evolution_compare.png"), dpi=300)
    plt.close()

    # assymetry series
    def asymmetry_series(Mb, method):
        w_star, _ = adaptive_weight_search(Mb, method)
        sc = fuse_scores(Mb, method, w_star[1], w_star[2], w_star[3], w_star[0])
        Mb2 = Mb.assign(score=sc)
        rows = []
        for (w0, w1), sub in Mb2.groupby(["win_start", "win_end"]):
            s01 = sub[(sub["src"] == 0) & (sub["tgt"] == 1)]["score"].sum()
            s10 = sub[(sub["src"] == 1) & (sub["tgt"] == 0)]["score"].sum()
            rows.append(
                {
                    "win_end": int(w1),
                    "delta_0to1_minus_1to0": float(s01 - s10)
                }
            )
        return pd.DataFrame(rows)
    
    def summarise_lead_fraction(M, method, sim_id=1, out_dir=CSV_DIR):
        series = asymmetry_series(M, method=method)
        if series is None or series.empty:
            return

        d = series["delta_0to1_minus_1to0"].to_numpy()
        m = np.isfinite(d)
        d = d[m]
        if d.size == 0:
            return

        n = float(d.size)
        frac_0 = float((d > 0).mean())  # fraction of time favouring 0->1
        frac_1 = float((d < 0).mean())  # fraction of time favouring 1->0

        def ci_95(p, n_):
            if n_ <= 0:
                return (np.nan, np.nan)
            se = math.sqrt(max(p * (1.0 - p), 0.0) / n_)
            lo = max(0.0, p - 1.96 * se)
            hi = min(1.0, p + 1.96 * se)
            return (float(lo), float(hi))

        lo0, hi0 = ci_95(frac_0, n)
        lo1, hi1 = ci_95(frac_1, n)

        out = pd.DataFrame(
            [
                dict(
                    method=method,
                    sim_id=sim_id,
                    n_windows=int(n),
                    frac_0_leads=frac_0,
                    frac_0_ci_lo=lo0,
                    frac_0_ci_hi=hi0,
                    frac_1_leads=frac_1,
                    frac_1_ci_lo=lo1,
                    frac_1_ci_hi=hi1,
                )
            ]
        )

        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"sim{sim_id}_lead_fraction_{method}.csv")
        out.to_csv(fname, index=False)


    for meth in [
        "tsmi_only",
        "te_only",
        "gc_only",
        "krlflow_only",
        "tsmi_plus_krl",
        "te_plus_krl",
        "gc_plus_krl"
    ]:
        AS = asymmetry_series(M, meth)
        AS.to_csv(os.path.join(CSV_DIR, f"sim{sim_id}_asym_{meth}.csv"), index=False)
        plt.figure()
        plt.plot(AS["win_end"], AS["delta_0to1_minus_1to0"])
        plt.axhline(0, ls="--")
        plt.title(f"Asymmetry ({meth}) - sim {sim_id}  (>0 favors 0→1)")
        plt.xlabel("time (window end)")
        plt.ylabel("score(0->1)-score(1->0)")
        plt.tight_layout()
        plt.savefig(
            os.path.join(FIG_DIR, f"sim{sim_id}_asym_{meth}.png"), dpi=300
        )
        plt.close()

    # PR/ROC
    if "is_true" in M.columns and np.isfinite(M["is_true"]).any():
        log("Computing PR/ROC (with ground truth)...")
        for abl in [
            "krlflow_only",
            "tsmi_only",
            "te_only",
            "gc_only",
            "tsmi_plus_krl",
            "te_plus_krl",
            "gc_plus_krl",
            "tsmi_te",
            "tsmi_te_plus_krl"
        ]:
            w_star, _ = adaptive_weight_search(M, abl)
            sc = fuse_scores(M, abl, w_star[1], w_star[2], w_star[3], w_star[0])
            y = M["is_true"].values
            fpr, tpr, thr_roc, auc, prec, rec, thr_pr, aupr = roc_pr_curves(y, sc)

            thr_pr_aligned = np.full(len(prec), np.nan, dtype=float)
            if len(thr_pr) == len(prec) - 1:
                thr_pr_aligned[1:] = thr_pr
            elif len(thr_pr) == len(prec):
                thr_pr_aligned[:] = thr_pr
            else:
                thr_pr_aligned[: min(len(thr_pr), len(prec))] = thr_pr[: min(len(thr_pr), len(prec))]

            thr_roc_aligned = np.full(len(fpr), np.nan, dtype=float)
            if len(thr_roc) == len(fpr):
                thr_roc_aligned[:] = thr_roc
            else:
                thr_roc_aligned[: min(len(thr_roc), len(fpr))] = thr_roc[: min(len(thr_roc), len(fpr))]

            save_csv_for_figure(f"sim{sim_id}_PR_{abl}",pd.DataFrame({"recall": rec, "precision": prec, "threshold": thr_pr_aligned}))
            save_csv_for_figure(f"sim{sim_id}_ROC_{abl}",pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr_roc_aligned}))
            plt.figure()
            plt.plot(rec, prec)
            plt.title(f"Precision-Recall - sim {sim_id} - {abl} (AUPR={aupr:.3f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, f"sim{sim_id}_PR_{abl}.png"), dpi=300)
            plt.close()
            plt.figure()
            plt.plot(fpr, tpr)
            plt.title(f"ROC - sim {sim_id} - {abl} (AUC={auc:.3f})")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, f"sim{sim_id}_ROC_{abl}.png"), dpi=300)
            plt.close()

    summarize_ablation_metrics(M, sim_id=sim_id)
    gate_significance_analysis(M, sim_id=sim_id)
    phase_times["evaluate_seconds"] = phase_times.get("evaluate_seconds", time.time() - t_eval0)

    n_edges = int(len(M))
    n_windows = int(M[["win_start", "win_end"]].drop_duplicates().shape[0])

    gate_stats = {}
    if "gate" in M.columns:
        g = np.asarray(M["gate"], float)
        if np.isfinite(g).any():
            gate_stats = {
                "mean": float(np.nanmean(g)),
                "std": float(np.nanstd(g)),
                "min": float(np.nanmin(g)),
                "max": float(np.nanmax(g)),
                "q25": float(np.nanpercentile(g, 25.0)),
                "median": float(np.nanpercentile(g, 50.0)),
                "q75": float(np.nanpercentile(g, 75.0))
            }

    summary = {
        "use_idtxl": USE_IDTXL,
        "has_idtxl": HAS_IDTXL,
        "window_size": WINDOW_SIZE,
        "window_step": WINDOW_STEP,
        "lags": LAGS,
        "strides": STRIDES,
        "te_nbins": TE_NBINS,
        "gc_maxlags": GC_MAXLAGS,
        "taus": {
            "TAU_DIST": TAU_DIST,
            "TAU_ALIGN": TAU_ALIGN,
            "TAU_R2": TAU_R2,
            "TAU_LCONS": TAU_LCONS,
            "TAU_PSI_VAR": TAU_PSI_VAR,
        },
        "num_edges": n_edges,
        "num_windows": n_windows,
        "phase_times_seconds": phase_times,
        "component_times_seconds": {
            k: float(v) for k, v in TIME_STATS.items()
        },
        "gate_distribution": gate_stats,
        "hyperparameters_notes": (
            "All main hyperparameters are set via environment variables "
            "(KRL_WINDOW_SIZE, KRL_WINDOW_STEP, KRL_LAG_TEMP, KRL_RIDGE_ALPHA, "
            "KRL_TAU_DIST, KRL_TAU_ALIGN, KRL_TAU_R2, KRL_TAU_LCONS, KRL_TAU_PSI_VAR, "
            "KRL_POS_THETA_COLS, KRL_POS_RADIUS_COLS, KRL_HEADING_COLS, KRL_ALIGN_COLS)."
        ),
    }

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    methods_for_fraction = [
        "tsmi_only",
        "te_only",
        "gc_only",
        "krlflow_only",
        "tsmi_plus_krl",
        "te_plus_krl",
        "gc_plus_krl",
    ]

    for mname in methods_for_fraction:
        try:
            summarise_lead_fraction(M, method=mname, sim_id=sim_id, out_dir=CSV_DIR)
        except Exception as exc:
            print(f"[WARN] Failed to summarise lead fraction for {mname}: {exc}")

def main():
    t0 = time.time()
    log("Loading Golden Shiner CSV...")
    D = load_gshiner_pair(CSV_IN)
    t1 = time.time()

    log("Computing edges...")
    M = compute_edges_gshiner(D)
    t2 = time.time()

    print(M["te"].describe())
    print(M["te"].value_counts().head())

    log("Attaching ground truth (if available)...")
    M = attach_labels_if_available(M)
    t3 = time.time()

    print(M["te"].describe())
    print(M["te"].value_counts().head())

    phase_times = {
        "load_seconds": t1 - t0,
        "compute_edges_seconds": t2 - t1,
        "attach_labels_seconds": t3 - t2
    }

    log("Evaluating and writing outputs...")
    evaluate_and_write(M, phase_times=phase_times)
    log("Done. Outputs at: " + OUT_DIR)

if __name__ == "__main__":
    main()
