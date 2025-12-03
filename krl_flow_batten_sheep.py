import os
import sys
import json
import math
import time
import random
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,roc_curve
from scipy.stats import zscore as _zscore_scipy

from BaseCode.ToProcessCode import createProcessList
from BaseCode.TrnsfmDataCode import transformData
from BaseCode.socialNetworkCode import cObsCounts, cCntrPts
from BaseCode.socialNetworkCode import snnmSigTest, snmSigTest
from BaseCode.DictionaryCode import cFNDict, cDTDict, cFBDict, cIFTDict

DATA_DIR = 'Data/NZ1FieldTripExperiment/'
SNN_FILE = 'nz1snmData.npy'
IFT_FILE = 'IFTFile.txt'
DT_FILE = 'DTFile.txt'
FN_FILE = 'FNFile.txt'
FB_FILE = 'FBFile.txt'
TP_FILE = 'ToProcess.txt'

OUT_DIR = 'results_krl_out_batten'
CSV_DIR = os.path.join(OUT_DIR, 'csv')
FIG_DIR = os.path.join(OUT_DIR, 'figs')
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

HAS_SK = HAS_STATSMODELS = HAS_IDTXL = False
USE_IDTXL = os.environ.get("KRL_USE_IDTXL", "1") == "1"

try:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAS_SK = True
except Exception:
    pass

try:
    from statsmodels.api import OLS, add_constant
    HAS_STATSMODELS = True
except Exception:
    pass

HAS_IDTXL = True
try:
    from IDTxl.idtxl.estimator import get_estimator
except Exception:
    HAS_IDTXL = False
    

RANDOM_SEED = 7
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

RUN_START = time.time()

WINDOW_SIZE = int(os.environ.get("KRL_WINDOW_SIZE", "3600"))
WINDOW_STEP = int(os.environ.get("KRL_WINDOW_STEP", "1800"))

BLK_SIZE1 = 6 * 3600
BLK_SIZE2 = 600

STRIDES = [1, 2]
LAGS = [1, 2, 3, 4]
TE_NBINS = [6, 8, 12] 
GC_MAXLAGS = [2, 4]

# Gates (base thresholds)
TAU_DIST = float(os.environ.get("KRL_TAU_DIST", "35.0"))
TAU_ALIGN = float(os.environ.get("KRL_TAU_ALIGN", "0.25"))
TAU_R2 = float(os.environ.get("KRL_TAU_R2", "0.01"))
TAU_LCONS = float(os.environ.get("KRL_TAU_LCONS", "0.4"))

# New gate knobs (stricter)
TAU_COLOC = float(os.environ.get("KRL_TAU_COLOC", "0.35"))
TAU_SVAR = float(os.environ.get("KRL_TAU_SVAR",  "0.45"))

LAG_TEMP = float(os.environ.get("KRL_LAG_TEMP", "0.5"))
RIDGE_ALPHA = float(os.environ.get("KRL_RIDGE_ALPHA", "1.0"))

N_SURR = int(os.environ.get("KRL_SURR", "30"))

# Utils
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def save_csv_for_figure(name: str, df: pd.DataFrame) -> None:
    try:
        if df is None:
            return
        safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(name))
        out_path = os.path.join(CSV_DIR, f"{safe}.csv")
        df.to_csv(out_path, index=False)
    except Exception as e:
        try:
            log(f"[WARN] Could not save CSV for figure '{name}': {e}")
        except Exception:
            pass


def zscore(v):
    v = np.asarray(v, float)
    if np.all(np.isnan(v)):
        return np.zeros_like(v)
    mu, sd = np.nanmean(v), np.nanstd(v)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(v)
    return (v - mu) / (sd + 1e-12)


def window_indices(T, win, step):
    s = 0
    while s < T:
        e = min(s + win, T)
        yield s, e
        if e == T:
            break
        s += step


# Geometry/Kinematics 
def compute_features_df(x):
    tPts = x.shape[0]
    nA = x.shape[2]
    rows = []
    for a in range(nA):
        xa = x[:, 0, a]
        ya = x[:, 1, a]
        dx = np.r_[np.nan, np.diff(xa)]
        dy = np.r_[np.nan, np.diff(ya)]
        heading = (np.arctan2(dy, dx) + np.pi) % (2 * np.pi) - np.pi
        turn = np.r_[np.nan, np.diff(heading)]
        turn = (turn + np.pi) % (2 * np.pi) - np.pi
        speed = np.hypot(dx, dy)
        accel = np.r_[np.nan, np.diff(speed)]
        for t in range(tPts):
            rows.append((a,t,xa[t],ya[t],heading[t],turn[t],speed[t],accel[t]))
    return (pd.DataFrame(rows,columns=["agent_id","t","x","y","heading","turn_rate","speed","accel"])
        .sort_values(["t", "agent_id"])
        .reset_index(drop=True))


def pairwise_median_distance(X, Y, w0, w1):
    a = X[w0:w1]
    b = Y[w0:w1]
    n = min(len(a), len(b))
    if n == 0:
        return np.nan
    return float(np.median(np.linalg.norm(a[:n] - b[:n], axis=1)))


def coloc_fraction(X, Y, Hx, Hy, w0, w1, tau_d=TAU_DIST, tau_a=TAU_ALIGN):
    n = min(w1 - w0, len(Hx) - w0, len(Hy) - w0)
    if n <= 5:
        return 0.0
    dx = X[w0:w0 + n, 0] - Y[w0:w0 + n, 0]
    dy = X[w0:w0 + n, 1] - Y[w0:w0 + n, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    align = np.cos(Hy[w0:w0 + n] - Hx[w0:w0 + n])
    return float(np.mean((dist < tau_d) & (align > tau_a)))


def var_compatibility(vx, vy, w0, w1):
    segx = vx[w0:w1]
    segy = vy[w0:w1]
    sx = np.nanstd(segx)
    sy = np.nanstd(segy)
    if (not np.isfinite(sx) or not np.isfinite(sy) or (sx + sy) <= 1e-9):
        return 0.0
    return float(1.0 - abs(sx - sy) / max(1e-9, sx + sy))


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


def _build_src_kinematics(src_head, src_turn, src_speed, src_accel, lag):
    sh = pd.Series(src_head).shift(lag).values
    tr = pd.Series(src_turn).shift(lag).values
    sp = pd.Series(src_speed).shift(lag).values
    ac = pd.Series(src_accel).shift(lag).values
    return np.column_stack([np.sin(sh),np.cos(sh),np.sin(tr),np.cos(tr),zscore(sp),zscore(ac)])


def _build_relational(src_df, tgt_df, lag):
    sh = pd.Series(src_df["heading"].values).shift(lag).values
    sp = pd.Series(src_df["speed"].values).shift(lag).values
    sx = pd.Series(src_df["x"].values).shift(lag).values
    sy = pd.Series(src_df["y"].values).shift(lag).values

    th = tgt_df["heading"].values
    tx = tgt_df["x"].values
    ty = tgt_df["y"].values
    tv = tgt_df["speed"].values

    dx = tx - sx
    dy = ty - sy
    dist = np.sqrt(dx**2 + dy**2)
    bearing = np.arctan2(dy, dx)
    align = np.cos(th - sh)
    dv = tv - sp
    ratio = np.divide(tv, np.where(np.abs(sp) < 1e-6, np.nan, sp))

    return np.column_stack([zscore(dist),np.sin(bearing),np.cos(bearing),align,zscore(dv),zscore(ratio)])


def krl_components(src_df, tgt_df, lags, temp):
    if not HAS_SK:
        return np.nan, [], [], {}

    tgt_head = tgt_df["heading"].values
    src_head = src_df["heading"].values
    src_turn = src_df["turn_rate"].values
    src_speed = src_df["speed"].values
    src_acc = src_df["accel"].values

    Xa, y = _build_auto_target_matrix(tgt_head)
    m0 = ~np.isnan(y) & ~np.isnan(Xa).any(1)
    if m0.sum() < 30:
        return np.nan, [], [], {}
    _, yhat_a = _fit_ridge(Xa[m0], y[m0])
    if yhat_a is None:
        return np.nan, [], [], {}
    ss_res_a = float(np.sum((y[m0] - yhat_a) ** 2))
    ss_tot = float(np.sum((y[m0] - np.nanmean(y[m0])) ** 2) + 1e-9)
    r2_auto = max(0.0, 1.0 - ss_res_a / ss_tot)

    gains = []
    for lag in lags:
        Xk = _build_src_kinematics(src_head, src_turn, src_speed, src_acc, lag)
        Xr = _build_relational(src_df, tgt_df, lag)
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
        w /= (w.sum() + 1e-12)
    lag_cons = (float(np.sort(w)[-2:].sum()) if w.size > 1 else float(w.max()) if w.size else np.nan)

    return (r2_auto,list(g),list(w),{"r2_auto": r2_auto,"r2_gain_max": float(np.nanmax(g) if len(g) else np.nan),"lag_consistency": lag_cons})


def tsmi_conditional_discrete(src_sig, tgt_sig, lag, z_feat, nbins=8):
    y_t = pd.Series(tgt_sig).values
    x_l = pd.Series(src_sig).shift(lag).values
    M = np.column_stack([y_t, x_l, z_feat])
    mask = ~np.isnan(M).any(1)
    M = M[mask]
    if len(M) < 160:
        return np.nan

    def qbin(v, b):
        finite = np.isfinite(v)
        if finite.sum() < b:
            return None
        edges = np.unique(
            np.quantile(v[finite], np.linspace(0, 1, b + 1))
        )
        if len(edges) < 3:
            return None
        out = np.digitize(v, edges[1:-1], right=True)
        out[~finite] = 0
        return out.astype(int)

    Y = qbin(M[:, 0], nbins)
    Xl = qbin(M[:, 1], nbins)
    if Y is None or Xl is None:
        return np.nan

    Zs = []
    for j in range(2, M.shape[1]):
        zz = qbin(M[:, j], nbins)
        if zz is not None:
            Zs.append(zz)
    if len(Zs) == 0:
        Zs = [np.zeros_like(Y, int)]

    c_y_x_z = defaultdict(float)
    c_x_z = defaultdict(float)
    c_y_z = defaultdict(float)
    c_z = defaultdict(float)
    N = len(Y)
    for i in range(N):
        zt = tuple(z[i] for z in Zs)
        c_y_x_z[(Y[i], Xl[i], zt)] += 1.0
        c_x_z[(Xl[i], zt)] += 1.0
        c_y_z[(Y[i], zt)] += 1.0
        c_z[zt] += 1.0

    total = float(N) + 1e-12
    cmi = 0.0
    for (yy, xx, zz), c in c_y_x_z.items():
        p_yx_z = c / total
        p_x_z = c_x_z[(xx, zz)] / total
        p_y_z = c_y_z[(yy, zz)] / total
        p_z = c_z[zz] / total
        if p_x_z > 0 and p_y_z > 0 and p_z > 0:
            cmi += p_yx_z * math.log((p_yx_z / p_x_z) / (p_y_z / p_z) + 1e-12)
    return float(max(cmi, 0.0))


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
        # só Y_{t-1}
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

def gc_core_arx_partial(src_sig, tgt_sig, lag, z_feat, maxlag=4):
    if not HAS_STATSMODELS:
        return np.nan
    y = pd.Series(tgt_sig).values
    x = pd.Series(src_sig).values
    p = int(maxlag)

    def lagmat(v, P):
        cols = []
        for k in range(1, P + 1):
            cols.append(pd.Series(v).shift(k).values)
        if P > 0:
            return np.column_stack(cols)
        return np.empty((len(v), 0))

    Ylags = lagmat(y, p)
    Xlags = lagmat(x, p)
    XlagL = pd.Series(x).shift(lag).values.reshape(-1, 1)

    M_full = np.column_stack([Ylags, XlagL, Xlags, z_feat])
    M_null = np.column_stack([Ylags, Xlags, z_feat])
    mask = ~np.isnan(M_full).any(1) & ~np.isnan(y)
    if mask.sum() < 5 * p + 60:
        return np.nan

    y = y[mask]
    Xf = M_full[mask]
    X0 = M_null[mask]
    try:
        m0 = OLS(y, add_constant(X0)).fit()
        m1 = OLS(y, add_constant(Xf)).fit()
        df1 = m1.df_model - m0.df_model
        df2 = m1.df_resid
        if df1 <= 0 or df2 <= 0:
            return np.nan
        RSS0, RSS1 = m0.ssr, m1.ssr
        F = ((RSS0 - RSS1) / df1) / (RSS1 / df2 + 1e-12)
        pval = math.exp(-max(F, 0.0))
        return float(1.0 - min(max(pval, 0.0), 1.0))
    except Exception:
        return np.nan


def _gate_base(dist_med, align_mean, r2_gain_max, lag_consistency):
    x = np.array([
            (TAU_DIST - dist_med) / max(1.0, TAU_DIST),
            (align_mean - TAU_ALIGN) / max(1e-6, abs(TAU_ALIGN)),
            (r2_gain_max - TAU_R2) / max(1e-6, abs(TAU_R2)),
            (lag_consistency - TAU_LCONS) / max(1e-6, abs(TAU_LCONS)),
        ],
        float,
    )
    w = np.array([0.6, 1.2, 1.0, 1.0])
    z = float(np.dot(w, x))
    return float(1.0 / (1.0 + math.exp(-z)))


def compute_group_metrics(df, XY, healthy_ids, sick_ids, w0, w1):
    agents = sorted(df["agent_id"].unique().tolist())
    A = {a: df[df["agent_id"] == a].reset_index(drop=True) for a in agents}

    med_dist = {}
    align = {}
    coloc = {}
    svar = {}
    for s in agents:
        for t in agents:
            if s == t:
                continue
            med_dist[(s, t)] = pairwise_median_distance(XY[s], XY[t], w0, w1)
            align[(s, t)] = np.nanmean(np.cos(A[t]["heading"].values[w0:w1] - A[s]["heading"].values[w0:w1]))
            coloc[(s, t)] = coloc_fraction(XY[s],XY[t],A[s]["heading"].values,A[t]["heading"].values,w0,w1)
            svar[(s, t)] = var_compatibility(A[s]["speed"].values, A[t]["speed"].values, w0, w1)

    rows = []
    for t in agents:
        for s in agents:
            if s == t:
                continue
            xs = A[s].iloc[w0:w1]
            xt = A[t].iloc[w0:w1]

            r2_auto, gains, weights, ginfo = krl_components(xs, xt, lags=LAGS, temp=LAG_TEMP)
            if not gains:
                gate = 0.0
                lag_star = LAGS[0]
                c_krl = np.nan
                c_tsmi = c_te = c_gc = np.nan
            else:
                g = np.nan_to_num(np.asarray(gains, float), 0.0)
                w = np.nan_to_num(np.asarray(weights, float), 0.0)
                c_krl = float((g * w).sum())
                lag_star = int(LAGS[int(np.argmax(w))])

                base_gate = _gate_base(
                    dist_med=med_dist[(s, t)],
                    align_mean=align[(s, t)],
                    r2_gain_max=float(np.nanmax(g)),
                    lag_consistency=ginfo.get(
                        "lag_consistency", 0.0
                    )
                )
                is_inter = int(((s in healthy_ids) and (t in sick_ids)) or ((s in sick_ids) and (t in healthy_ids)))
                g_coloc = coloc[(s, t)]
                g_svar = svar[(s, t)]
                inter_penalty = (1.0 - 0.5 * is_inter * (1.0 - g_coloc)) * (0.5 + 0.5 * max(0.0, g_svar))
                gate = float(np.clip(base_gate * inter_penalty, 0.0, 1.0))

                Xk = _build_src_kinematics(
                    xs["heading"].values,
                    xs["turn_rate"].values,
                    xs["speed"].values,
                    xs["accel"].values,
                    lag_star
                )
                Xr = _build_relational(xs, xt, lag_star)
                Z = np.nan_to_num(np.column_stack([Xk, Xr]), 0.0)
                Yh = xt["heading"].values
                nbins_te = select_te_nbins_from_signal(Yh, TE_NBINS)

                # TSMI 
                c_tsmi = tsmi_conditional_discrete(
                    xs["heading"].values,
                    xt["heading"].values,
                    lag=lag_star,
                    z_feat=Z,
                    nbins=nbins_te
                )

                # TE 
                if HAS_IDTXL and USE_IDTXL:
                    try:
                        c_te = te_core_partial_idtxl(
                            xs["heading"].values,
                            xt["heading"].values,
                            lag=lag_star,
                            z_feat=Z,
                            nbins=nbins_te,
                        )
                    except Exception:
                        c_te = np.nan
                else:
                    c_te = np.nan

                # GC
                c_gc = np.nan
                for gl in GC_MAXLAGS:
                    v = gc_core_arx_partial(
                        xs["heading"].values,
                        xt["heading"].values,
                        lag=lag_star,
                        z_feat=Z,
                        maxlag=gl
                    )
                    if np.isfinite(v):
                        c_gc = v
                        break

            rows.append(
                {
                    "win_start": w0,
                    "win_end": w1,
                    "src": s,
                    "tgt": t,
                    "tsmi": c_tsmi,
                    "te": c_te,
                    "gc": c_gc,
                    "krl": c_krl,
                    "gate": gate,
                    "dist": med_dist[(s, t)],
                    "align": align[(s, t)],
                    "coloc": coloc[(s, t)],
                    "svar": svar[(s, t)],
                    "src_group": ("H" if s in healthy_ids else "S"),
                    "tgt_group": ("H" if t in healthy_ids else "S")
                }
            )

    M = pd.DataFrame(rows)

    def fuse(base, krl, gate, coloc, svar, lam=0.6):
        base = np.nan_to_num(base, 0.0)
        k = np.nan_to_num(krl, 0.0)
        cap = (np.clip(gate, 0, 1) ** 1.5
            * (0.5 + 0.5 * np.clip(coloc, 0, 1))
            * (0.5 + 0.5 * np.clip(svar, 0, 1)))
        u = np.tanh(np.abs(base))
        beta = np.clip(lam * cap * (1 - u), 0.0, 0.8)
        return (1 - beta) * base + beta * k

    for met in ["tsmi", "te", "gc"]:
        M[f"{met}_krl"] = fuse(
            M[met].values,
            M["krl"].values,
            M["gate"].values,
            M["coloc"].values,
            M["svar"].values
        )

    def agg(sel):
        return M.loc[
            sel,
            [
                "tsmi",
                "te",
                "gc",
                "tsmi_krl",
                "te_krl",
                "gc_krl"
            ]
        ].mean(numeric_only=True)

    intra_H = agg((M.src_group == "H") & (M.tgt_group == "H"))
    intra_S = agg((M.src_group == "S") & (M.tgt_group == "S"))
    inter = agg(((M.src_group == "H") & (M.tgt_group == "S")) | ((M.src_group == "S") & (M.tgt_group == "H")))

    org = (
        pd.DataFrame([intra_H, intra_S, inter],index=["intra_H", "intra_S", "inter"])
        .reset_index()
        .rename(columns={"index": "pair_class"})
    )
    return M, org


def circular_shift_surrogate(arr, max_shift=None):
    n = len(arr)
    if n <= 3:
        return arr.copy()
    if max_shift is None:
        max_shift = n // 2
    k = random.randint(1, max(1, min(max_shift, n - 1)))
    return np.r_[arr[k:], arr[:k]]


def make_surrogates_group(df, n_surr=30):
    S = {}
    for aid, sub in df.groupby("agent_id"):
        sub = sub.sort_values("t")
        sh = sub["heading"].values
        su = circular_shift_surrogate(sh)
        S[aid] = sub.copy()
        S[aid]["heading"] = su
    return (pd.concat(S.values(), ignore_index=True).sort_values(["t", "agent_id"]).reset_index(drop=True))


def block_modularity_entropy_MI(etrajMat, XC, nC, blk_size1, blk_size2):
    from sklearn.metrics import mutual_info_score
    from BaseCode.socialNetworkCode import cMarkovModel

    nX, T, nA = etrajMat.shape
    nBlks = T // blk_size1 if T >= blk_size1 else 1
    AdjVals = []
    MIs = []
    Mods = []
    for blk in range(nBlks):
        s = blk * blk_size1
        e = min(T, (blk + 1) * blk_size1)
        ccp, ccpc = cObsCounts(etrajMat[:, s:e, :], XC, nC, blk_size2)
        mi = []
        for i in range(nA):
            for j in range(i + 1, nA):
                mi.append(mutual_info_score(ccp[:, i], ccp[:, j]))
        MIs.append(np.nanmean(mi) if len(mi) else np.nan)

        coc = np.zeros((nA, nA))
        for i in range(nA):
            for j in range(i + 1, nA):
                coc[i, j] = np.sum(ccp[:, i] == ccp[:, j])
                coc[j, i] = coc[i, j]

        p0, tMat = cMarkovModel(
            ccp, nC, nA, blk_size1 / blk_size2, iid=True
        )
        Adj = snmSigTest(p0,tMat,coc,0.005,nA,blk_size1 / blk_size2,nC,1)
        AdjVals.append(np.mean(Adj))

        try:
            from scipy.sparse import csgraph

            lap = csgraph.laplacian(Adj, normed=False)
            eigvals, eigvecs = np.linalg.eigh(lap)
            idx = np.argsort(eigvals)
            if len(idx) > 1:
                v2 = eigvecs[:, idx[1]]
            else:
                v2 = eigvecs[:, idx[0]]
            part = (v2 >= 0).astype(int)
            m = Adj.sum() / 2.0 + 1e-9
            deg = Adj.sum(1)
            Q = 0.0
            for i in range(nA):
                for j in range(nA):
                    if part[i] == part[j]:
                        Q += Adj[i, j] - (deg[i] * deg[j]) / (2 * m)
            Q /= (2 * m)
            Mods.append(Q)
        except Exception:
            Mods.append(np.nan)

    return (np.nanmean(AdjVals),np.nanmean(MIs),np.nanmean(Mods))

def process_day_file(fName, fbDict, iftDict, dtDict, fnDict):
    data = np.load(os.path.join(DATA_DIR, fName + ".npy"),allow_pickle=True,encoding="latin1",)
    timeMat, obs_indicator, LSG, trajMat = data
    healthy_idx = np.where(LSG[:, 4] == 5)[0]
    sick_idx = np.where(LSG[:, 4] == 6)[0]

    sTime = datetime(*timeMat[0])
    eTime = datetime(*timeMat[1])
    ifsTime, ifeTime = iftDict[fName]
    sIndex = (ifsTime - sTime).seconds + 86400 * (ifsTime - sTime).days
    eIndex = (ifeTime - sTime).seconds + 86400 * (ifeTime - sTime).days

    etraj_H = trajMat[:, sIndex:eIndex, :][:, :, healthy_idx]
    etraj_S = trajMat[:, sIndex:eIndex, :][:, :, sick_idx]

    fn = fnDict[fName]
    dataTransform = dtDict[fn]
    etraj_H = transformData(etraj_H, *dataTransform[0])
    etraj_S = transformData(etraj_S, *dataTransform[0])

    fb = fbDict[fn]
    XC, _, _ = cCntrPts(fb, 5, 5)

    def stack_groups(eH, eS):
        X = np.concatenate([eH, eS], axis=2)
        df = compute_features_df(np.transpose(X, (1, 0, 2)))
        nH = eH.shape[2]
        agents = sorted(df["agent_id"].unique().tolist())
        mapH = {a for a in agents if a < nH}
        mapS = {a for a in agents if a >= nH}
        XY = {a: df[df["agent_id"] == a][["x", "y"]].to_numpy() for a in agents}
        return df, XY, mapH, mapS

    df_all, XY, H_ids, S_ids = stack_groups(etraj_H, etraj_S)
    T = int(df_all["t"].max()) + 1

    ORG_ROWS = []
    EDGE_ROWS = []
    for w0, w1 in window_indices(T, WINDOW_SIZE, WINDOW_STEP):
        M_edges, org = compute_group_metrics(df_all, XY, H_ids, S_ids, w0, w1)
        M_edges["day_file"] = fName
        org["day_file"] = fName
        ORG_ROWS.append(org)
        EDGE_ROWS.append(M_edges)

    ORG = pd.concat(ORG_ROWS, ignore_index=True) if ORG_ROWS else pd.DataFrame()
    EDG = pd.concat(EDGE_ROWS, ignore_index=True) if EDGE_ROWS else pd.DataFrame()

    adj_H, mi_H, mod_H = block_modularity_entropy_MI(etraj_H, XC, 25, BLK_SIZE1, BLK_SIZE2)
    adj_S, mi_S, mod_S = block_modularity_entropy_MI(etraj_S, XC, 25, BLK_SIZE1, BLK_SIZE2)

    SURR = []
    for si in range(N_SURR):
        df_surr = make_surrogates_group(df_all)
        agents = sorted(df_surr["agent_id"].unique().tolist())
        XYs = {a: df_surr[df_surr["agent_id"] == a][["x", "y"]].to_numpy() for a in agents}
        rows = []
        for w0, w1 in window_indices(T, WINDOW_SIZE, WINDOW_STEP):
            _, org_s = compute_group_metrics(df_surr, XYs, H_ids, S_ids, w0, w1)
            org_s["surr_id"] = si
            rows.append(org_s)
        SURR.append(pd.concat(rows, ignore_index=True))
    SURR = pd.concat(SURR, ignore_index=True) if len(SURR) else pd.DataFrame()

    return EDG,ORG,{"adj_H": adj_H,"mi_H": mi_H,"mod_H": mod_H,"adj_S": adj_S,"mi_S": mi_S,"mod_S": mod_S},H_ids,S_ids


def safe_auc(y, s):
    try:
        if len(np.unique(y)) < 2:
            return np.nan
        return float(roc_auc_score(y, s))
    except Exception:
        return np.nan


def cliffs_delta(a, b, max_samples=1000):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan

    if len(a) > max_samples:
        a = np.random.choice(a, size=max_samples, replace=False)
    if len(b) > max_samples:
        b = np.random.choice(b, size=max_samples, replace=False)

    n, m = len(a), len(b)
    gt = 0
    lt = 0
    for x in a:
        gt += np.sum(x > b)
        lt += np.sum(x < b)
    return float((gt - lt) / (n * m))


def gate_significance_analysis(df_edges, metric_cols, q_low=0.33, q_high=0.66, B=400):
    gate = df_edges["gate"].to_numpy()
    g_low = np.nanpercentile(gate, q_low * 100.0)
    g_high = np.nanpercentile(gate, q_high * 100.0)

    low_mask = gate <= g_low
    high_mask = gate >= g_high

    res = []
    for metric in metric_cols:
        v = df_edges[metric].to_numpy()
        a = v[high_mask]
        b = v[low_mask]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        n1, n0 = len(a), len(b)
        if n1 < 20 or n0 < 20:
            continue

        mean_high = float(np.nanmean(a))
        mean_low = float(np.nanmean(b))
        diff = mean_high - mean_low
        cd = cliffs_delta(a, b)

        if B > 0:
            pool = np.concatenate([a, b])
            n = len(pool)
            idx = np.arange(n)
            cnt = 0
            for _ in range(B):
                np.random.shuffle(idx)
                a_p = pool[idx[:n1]]
                b_p = pool[idx[n1:]]
                d_p = np.nanmean(a_p) - np.nanmean(b_p)
                if abs(d_p) >= abs(diff):
                    cnt += 1
            p_perm = (cnt + 1.0) / (B + 1.0)
        else:
            p_perm = np.nan

        res.append(
            {
                "metric": metric,
                "gate_low_thr": float(g_low),
                "gate_high_thr": float(g_high),
                "n_low": int(n0),
                "n_high": int(n1),
                "mean_low": mean_low,
                "mean_high": mean_high,
                "delta_high_minus_low": diff,
                "cliffs_delta": float(cd)
                if np.isfinite(cd)
                else np.nan,
                "p_perm_mean_diff": float(p_perm),
            }
        )

    return pd.DataFrame(res)

def zscore_series(x):
    x = np.asarray(x, float)
    x = np.where(np.isfinite(x), x, np.nan)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x)
    z = (x - mu) / (sd + 1e-12)
    z[~np.isfinite(z)] = 0.0
    return z


def fuse_scores(df_edges, cols):
    if not cols:
        return np.zeros(len(df_edges), float)
    zs = [zscore_series(df_edges[c].to_numpy()) for c in cols]
    return np.nan_to_num(np.mean(zs, axis=0), nan=0.0)


def main():
    log("KRL-Flow Batten (vKRLplus) - start")

    fbDict = cFBDict(FB_FILE)
    iftDict = cIFTDict(IFT_FILE)
    dtDict = cDTDict(DT_FILE)
    fnDict = cFNDict(FN_FILE)
    ftProcess = createProcessList(TP_FILE)

    ALL_EDGES = []
    ALL_ORG = []
    ALL_DAYSTATS = []
    day_idx = 0
    for fName in ftProcess:
        day_idx += 1
        log(f">> Processing {fName} (day {day_idx})")
        EDG, ORG, STATS, H_ids, S_ids = process_day_file(fName, fbDict, iftDict, dtDict, fnDict)

        EDG.to_csv(os.path.join(CSV_DIR, f"edges_{day_idx:02d}_{fName}.csv"),index=False)
        ORG.to_csv(os.path.join(CSV_DIR, f"org_{day_idx:02d}_{fName}.csv"),index=False)
        with open(os.path.join(CSV_DIR, f"stats_{day_idx:02d}_{fName}.json"),"w") as f:
            json.dump(STATS, f, indent=2)

        EDG["day"] = day_idx
        ORG["day"] = day_idx
        ALL_EDGES.append(EDG)
        ALL_ORG.append(ORG)
        STATS["day"] = day_idx
        STATS["file"] = fName
        ALL_DAYSTATS.append(STATS)

    if not len(ALL_EDGES):
        raise FileNotFoundError("No .npy files found from ToProcess.txt/dictionaries.")

    DF_E = pd.concat(ALL_EDGES, ignore_index=True)
    DF_O = pd.concat(ALL_ORG, ignore_index=True)
    DF_D = pd.DataFrame(ALL_DAYSTATS)

    mets = ["tsmi", "te", "gc", "tsmi_krl", "te_krl", "gc_krl"]
    piv = DF_O.pivot_table(index=["day", "pair_class"], values=mets, aggfunc="mean").reset_index()

    def plot_series(metric, title):
        plt.figure(figsize=(8.6, 4))
        if metric in piv.columns:
            _df_fig = piv[["day", "pair_class", metric]].copy()
            save_csv_for_figure(f"series_{metric}", _df_fig)
        for cls, sub in piv.groupby("pair_class"):
            plt.plot(sub["day"],sub[metric],marker="o",label=cls)
        plt.axvline(4, linestyle="--", alpha=0.7)
        plt.xlabel("Day")
        plt.ylabel(metric.upper())
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"series_{metric}.png"),dpi=300)
        plt.close()

    for m in mets:
        plot_series(m, f"Per-group evolution - {m}")

    def day_delta(metric, cls):
        pre = piv[(piv["day"] <= 3) & (piv["pair_class"] == cls)][metric].mean()
        post = piv[(piv["day"] >= 4) & (piv["pair_class"] == cls)][metric].mean()
        return float(post - pre)

    DELTAS = []
    for m in mets:
        for cls in ["intra_H", "intra_S", "inter"]:
            DELTAS.append(
                {
                    "metric": m,
                    "pair_class": cls,
                    "delta_after_day4": day_delta(m, cls)
                }
            )
    DF_DELTA = pd.DataFrame(DELTAS)
    DF_DELTA.to_csv(os.path.join(CSV_DIR, "deltas_after_day4.csv"),index=False)

    DF_E["is_intra"] = ((DF_E["src_group"] == DF_E["tgt_group"]).astype(int))
    Y = DF_E["is_intra"].values

    PRROC = []

    def prroc(y, s, tag):
        m = np.isfinite(s)
        if m.sum() < 5 or len(np.unique(y[m])) < 2:
            return
        prec, rec, thr_pr = precision_recall_curve(y[m], s[m])
        fpr, tpr, thr_roc = roc_curve(y[m], s[m])
        aupr = float(average_precision_score(y[m], s[m]))
        try:
            _df_pr = pd.DataFrame({
                "recall": rec,
                "precision": prec,
                "threshold": np.r_[np.nan, thr_pr],
            })
            save_csv_for_figure(f"PR_{tag}", _df_pr)

            _df_roc = pd.DataFrame({
                "fpr": fpr,
                "tpr": tpr,
                "threshold": thr_roc,
            })
            save_csv_for_figure(f"ROC_{tag}", _df_roc)
        except Exception:
            pass
        plt.figure()
        plt.plot(rec, prec)
        plt.title(f"PR – {tag} (AUPR={aupr:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"PR_{tag}.png"), dpi=300)
        plt.close()

        plt.figure()
        plt.plot(fpr, tpr)
        plt.title(f"ROC - {tag} (AUC={safe_auc(y[m], s[m]):.3f})")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"ROC_{tag}.png"), dpi=300)
        plt.close()

    for base in ["tsmi", "te", "gc"]:
        sc_b = DF_E[base].values
        sc_k = DF_E[f"{base}_krl"].values
        prroc(Y, sc_b, f"{base}_baseline")
        prroc(Y, sc_k, f"{base}_plusKRL")
        PRROC.append({
                "metric": base,
                "AUROC_baseline": safe_auc(Y, sc_b),
                "AUROC_plusKRL": safe_auc(Y, sc_k)
            }
        )

    DF_PRROC = pd.DataFrame(PRROC)
    DF_PRROC.to_csv(os.path.join(CSV_DIR, "prroc_intra_vs_inter.csv"),index=False)

    DF_D.to_csv(os.path.join(CSV_DIR, "day_group_stats.jsonl"),index=False)

    def permutation_pvalue_auroc(y, s_base, s_krl, B=800):
        obs = safe_auc(y, s_krl) - safe_auc(y, s_base)
        m = np.isfinite(s_base) & np.isfinite(s_krl)
        y = y[m]
        sb = s_base[m]
        sk = s_krl[m]
        if len(np.unique(y)) < 2 or len(y) < 20:
            return np.nan
        cnt = 0
        for _ in range(B):
            mask = np.random.rand(len(sb)) < 0.5
            s1 = np.where(mask, sb, sk)
            s2 = np.where(mask, sk, sb)
            diff = safe_auc(y, s2) - safe_auc(y, s1)
            if diff >= obs:
                cnt += 1
        return (cnt + 1.0) / (B + 1.0)

    STATS = []
    for base in ["tsmi", "te", "gc"]:
        sb = DF_E[base].values
        sk = DF_E[f"{base}_krl"].values
        p = permutation_pvalue_auroc(Y, sb, sk, B=800)
        STATS.append(
            {
                "metric": base,
                "AUROC_base": safe_auc(Y, sb),
                "AUROC_plusKRL": safe_auc(Y, sk),
                "delta": safe_auc(Y, sk) - safe_auc(Y, sb),
                "p_perm": p
            }
        )
    DF_STATS = pd.DataFrame(STATS)
    DF_STATS.to_csv(os.path.join(CSV_DIR, "stats_perm_auroc_gain.csv"),index=False)

    gate_metrics = [
        "tsmi",
        "te",
        "gc",
        "tsmi_krl",
        "te_krl",
        "gc_krl",
        "krl",
        "dist",
        "align",
        "coloc",
        "svar",
    ]
    DF_GATE = gate_significance_analysis(DF_E, gate_metrics, q_low=0.33, q_high=0.66, B=400)
    DF_GATE.to_csv(os.path.join(CSV_DIR, "gate_significance_metrics.csv"),index=False)

    ABLATIONS = {
        # Baselines
        "tsmi_only": ["tsmi"],
        "te_only": ["te"],
        "gc_only": ["gc"],
        "tsmi_te": ["tsmi", "te"],
        "tsmi_te_gc": ["tsmi", "te", "gc"],
        "krl_only": ["krl"],
        # With KRL
        "tsmi_krl": ["tsmi_krl"],
        "te_krl": ["te_krl"],
        "gc_krl": ["gc_krl"],
        "tsmi_te_krl": ["tsmi_krl", "te_krl"],
        "tsmi_te_gc_krl": ["tsmi_krl", "te_krl", "gc_krl"]
    }

    ABL_STATS = []
    for name, cols in ABLATIONS.items():
        score = fuse_scores(DF_E, cols)
        DF_E[f"abl_{name}"] = score
        m = np.isfinite(score)
        if m.sum() < 20 or len(np.unique(Y[m])) < 2:
            auc = np.nan
            aupr = np.nan
        else:
            auc = safe_auc(Y[m], score[m])
            aupr = float(average_precision_score(Y[m], score[m]))
        ABL_STATS.append(
            {
                "ablation": name,
                "cols": ",".join(cols),
                "AUROC_intra_vs_inter": auc,
                "AUPR_intra_vs_inter": aupr
            }
        )
    DF_ABL = pd.DataFrame(ABL_STATS)
    DF_ABL.to_csv(os.path.join(CSV_DIR, "ablations_intra_vs_inter.csv"),index=False)

    for base in ["tsmi", "te", "gc"]:
        plt.figure(figsize=(9.2, 4))
        try:
            _df_fig = piv[piv["pair_class"].isin(["intra_H", "intra_S", "inter"])][["day", "pair_class", base, f"{base}_krl"]].copy()
            save_csv_for_figure(f"org_by_day_{base}", _df_fig)
        except Exception:
            pass
        for cls in ["intra_H", "intra_S", "inter"]:
            s1 = piv[piv["pair_class"] == cls][base]
            s2 = piv[piv["pair_class"] == cls][f"{base}_krl"]
            dd = piv[piv["pair_class"] == cls]["day"]
            plt.plot(dd,s1,marker="o",label=f"{cls}-{base}")
            plt.plot(dd,s2,marker="^",linestyle="--",label=f"{cls}-{base}+KRL")
        plt.axvline(4, linestyle="--", alpha=0.7)
        plt.title(f"Organization by day – {base} vs {base}+KRL")
        plt.xlabel("Day")
        plt.ylabel("mean score")
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"org_by_day_{base}.png"),dpi=300)
        plt.close()

    runtime_seconds = float(time.time() - RUN_START)
    unique_windows = (DF_E[["day", "win_start", "win_end"]].drop_duplicates().shape[0])

    HYPERPARAMS = {
        "WINDOW_SIZE": WINDOW_SIZE,
        "WINDOW_STEP": WINDOW_STEP,
        "BLK_SIZE1": BLK_SIZE1,
        "BLK_SIZE2": BLK_SIZE2,
        "STRIDES": STRIDES,
        "LAGS": LAGS,
        "TE_NBINS": TE_NBINS,
        "GC_MAXLAGS": GC_MAXLAGS,
        "TAU_DIST": TAU_DIST,
        "TAU_ALIGN": TAU_ALIGN,
        "TAU_R2": TAU_R2,
        "TAU_LCONS": TAU_LCONS,
        "TAU_COLOC": TAU_COLOC,
        "TAU_SVAR": TAU_SVAR,
        "LAG_TEMP": LAG_TEMP,
        "RIDGE_ALPHA": RIDGE_ALPHA,
        "N_SURR": N_SURR,
        "USE_IDTXL": USE_IDTXL,
    }

    GATE_STATS = {
        "gate_mean": float(DF_E["gate"].mean()),
        "gate_std": float(DF_E["gate"].std()),
        "gate_q05": float(DF_E["gate"].quantile(0.05)),
        "gate_q50": float(DF_E["gate"].quantile(0.50)),
        "gate_q95": float(DF_E["gate"].quantile(0.95)),
        "prop_gate_gt_0_5": float((DF_E["gate"] > 0.5).mean()),
        "prop_gate_gt_0_75": float((DF_E["gate"] > 0.75).mean())
    }

    summary = {
        "notes": (
            "Batten sheep - organization (intra/inter), Day-4 change, "
            "MI/Adj/Mod, +KRL ablation, gate significance, hyperparams & runtime."
        ),
        "PRROC_intra_vs_inter": PRROC,
        "perm_test_auroc_gain": STATS,
        "delta_after_day4": DF_DELTA.to_dict(orient="records"),
        "ablation_intra_vs_inter": ABL_STATS,
        "gate_significance": DF_GATE.to_dict(orient="records"),
        "hyperparameters": HYPERPARAMS,
        "gate_stats": GATE_STATS,
        "runtime_seconds": runtime_seconds,
        "n_edges": int(len(DF_E)),
        "n_windows": int(unique_windows)
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log("KRL-Flow Batten (vKRLplus) - done")


if __name__ == "__main__":
    main()
