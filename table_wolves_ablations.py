import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path("results_krl_out_wolves") 
N_SIMS = 10

ABLATIONS_PATTERN = "sim{sim_id}_ablations.csv"
BEST_PATTERN = "sim{sim_id}_best_by_ablation.csv"
STATS_PATTERN = "sim{sim_id}_stats.json"

METRIC_COLS = [
    "auroc",
    "aupr",
    "cliffs_delta",
    "prec_at_k",
    "recall_at_k",
    "f1_at_k",
    "hits_at_k",
    "ndcg_at_k",
]


def load_all_ablations():
    dfs = []
    for sim_id in range(1, N_SIMS + 1):
        path = DATA_DIR / "csv" / ABLATIONS_PATTERN.format(sim_id=sim_id)
        df = pd.read_csv(path)
        df["sim_id"] = sim_id
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def summarize_ablations(df):
    agg_dict = {m: ["mean", "std"] for m in METRIC_COLS if m in df.columns}

    grouped = df.groupby("ablation").agg(agg_dict)

    grouped.columns = [f"{m}_{stat}" for m, stat in grouped.columns.to_flat_index()]
    grouped = grouped.reset_index()
    return grouped


def load_all_best():
    dfs = []
    for sim_id in range(1, N_SIMS + 1):
        path = DATA_DIR / "csv" / BEST_PATTERN.format(sim_id=sim_id)
        df = pd.read_csv(path)
        df["sim_id"] = sim_id
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def summarize_best(best_df):
    if "auroc" not in best_df.columns:
        raise ValueError("Coluna 'auroc' não encontrada em best_by_ablation.")

    idx_best = best_df.groupby("sim_id")["auroc"].idxmax()
    winners = best_df.loc[idx_best, ["sim_id", "ablation"]]

    freq = (
        winners.groupby("ablation")
        .size()
        .reset_index(name="n_sims_best_auroc")
        .sort_values("n_sims_best_auroc", ascending=False)
    )
    return freq


def summarize_runtime():
    rows = []
    for sim_id in range(1, N_SIMS + 1):
        path = DATA_DIR / STATS_PATTERN.format(sim_id=sim_id)
        with open(path, "r") as f:
            stats = json.load(f)
        rows.append(
            {
                "sim_id": sim_id,
                "runtime_seconds": stats.get("runtime_seconds", float("nan")),
            }
        )

    df = pd.DataFrame(rows)
    summary = df["runtime_seconds"].agg(["mean", "std", "min", "max"]).to_frame().T
    return df, summary


def main():
    all_abl = load_all_ablations()
    abl_summary = summarize_ablations(all_abl)
    abl_summary.to_csv(DATA_DIR / "wolves_ablation_summary_mean_std.csv", index=False)

    best_df = load_all_best()
    best_freq = summarize_best(best_df)
    best_freq.to_csv(DATA_DIR / "wolves_best_ablation_freq.csv", index=False)

    runtime_df, runtime_summary = summarize_runtime()
    runtime_df.to_csv(DATA_DIR / "wolves_runtime_by_sim.csv", index=False)
    runtime_summary.to_csv(DATA_DIR / "wolves_runtime_summary.csv", index=False)

    print("Arquivos gerados em:", DATA_DIR.resolve())


if __name__ == "__main__":
    main()
