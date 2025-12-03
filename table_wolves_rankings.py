import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path("results_krl_out_wolves/csv")

LEADER_ID = 100
FOLLOWER_IDS = list(range(101, 115))
INDEPENDENT_IDS = list(range(115, 130))
TOP_K = 15

METHODS = [
    "tsmi_only",
    "tsmi_plus_krl",
    "te_only",
    "te_plus_krl",
    "gc_only",
    "gc_plus_krl",
    "krlflow_only",
]

def compute_group_ranking_metrics(df, method, sim_id, top_k=TOP_K):

    df_sorted = df.sort_values("score", ascending=False).reset_index(drop=True)
    df_sorted["rank"] = np.arange(1, len(df_sorted) + 1)

    def get_rank(agent_id):
        row = df_sorted.loc[df_sorted["src"] == agent_id, "rank"]
        return float(row.iloc[0]) if not row.empty else np.nan

    leader_rank = get_rank(LEADER_ID)

    follower_ranks = df_sorted.loc[df_sorted["src"].isin(FOLLOWER_IDS), "rank"].to_numpy()
    indep_ranks = df_sorted.loc[df_sorted["src"].isin(INDEPENDENT_IDS), "rank"].to_numpy()

    def mean_or_nan(arr):
        return float(np.mean(arr)) if arr.size > 0 else np.nan

    def std_or_nan(arr):
        return float(np.std(arr, ddof=0)) if arr.size > 0 else np.nan

    mean_followers = mean_or_nan(follower_ranks)
    std_followers = std_or_nan(follower_ranks)

    mean_indep = mean_or_nan(indep_ranks)
    std_indep = std_or_nan(indep_ranks)

    delta_FI = mean_indep - mean_followers if np.isfinite(mean_followers) and np.isfinite(mean_indep) else np.nan

    F_topk = int((follower_ranks <= top_k).sum()) if follower_ranks.size > 0 else 0
    I_topk = int((indep_ranks <= top_k).sum()) if indep_ranks.size > 0 else 0

    if np.isfinite(leader_rank) and follower_ranks.size > 0:
        prop_followers_below_leader = float((follower_ranks > leader_rank).sum() / follower_ranks.size)
    else:
        prop_followers_below_leader = np.nan

    if follower_ranks.size > 0 and indep_ranks.size > 0:
        fr = follower_ranks[:, None]
        ir = indep_ranks[None, :]
        correct_pairs = (ir > fr).sum()
        total_pairs = fr.size * ir.size
        prop_pairs_FI_ordered = float(correct_pairs / total_pairs)
    else:
        prop_pairs_FI_ordered = np.nan

    return {
        "simulation": sim_id,
        "method": method,
        "leader_rank": leader_rank,
        "mean_followers_rank": mean_followers,
        "std_followers_rank": std_followers,
        "mean_indep_rank": mean_indep,
        "std_indep_rank": std_indep,
        "delta_FI": delta_FI,
        f"followers_at_top{top_k}": F_topk,
        f"indep_at_top{top_k}": I_topk,
        "prop_followers_below_leader": prop_followers_below_leader,
        "prop_pairs_FI_ordered": prop_pairs_FI_ordered,
    }

records = []
NUM_SIMS = 10

for sim_id in range(1, NUM_SIMS + 1):
    for method in METHODS:
        path = BASE_DIR / f"sim{sim_id}_ranking_{method}.csv"
        if not path.exists():
            continue

        df = pd.read_csv(path)

        if not {"src", "score"}.issubset(df.columns):
            raise ValueError(
                f"Esperado colunas 'src' e 'score' em {path}, "
                f"mas encontrei {df.columns.tolist()}"
            )

        rec = compute_group_ranking_metrics(df, method, sim_id)
        records.append(rec)

per_sim_df = pd.DataFrame.from_records(records)

per_sim_csv = BASE_DIR / "group_ranking_per_simulation.csv"
per_sim_df.to_csv(per_sim_csv, index=False)
print(f"Arquivo salvo: {per_sim_csv}")

summary = per_sim_df.groupby("method").agg(
    leader_rank_mean=("leader_rank", "mean"),
    leader_rank_std=("leader_rank", "std"),
    mean_followers_rank_mean=("mean_followers_rank", "mean"),
    mean_followers_rank_std=("mean_followers_rank", "std"),
    mean_indep_rank_mean=("mean_indep_rank", "mean"),
    mean_indep_rank_std=("mean_indep_rank", "std"),
    delta_FI_mean=("delta_FI", "mean"),
    delta_FI_std=("delta_FI", "std"),
    followers_at_top15_mean=(f"followers_at_top{TOP_K}", "mean"),
    followers_at_top15_std=(f"followers_at_top{TOP_K}", "std"),
    indep_at_top15_mean=(f"indep_at_top{TOP_K}", "mean"),
    indep_at_top15_std=(f"indep_at_top{TOP_K}", "std"),
    prop_followers_below_leader_mean=("prop_followers_below_leader", "mean"),
    prop_followers_below_leader_std=("prop_followers_below_leader", "std"),
    prop_pairs_FI_ordered_mean=("prop_pairs_FI_ordered", "mean"),
    prop_pairs_FI_ordered_std=("prop_pairs_FI_ordered", "std")
).reset_index()

summary_csv = BASE_DIR / "group_ranking_summary_by_method.csv"
summary.to_csv(summary_csv, index=False)
print(f"Arquivo salvo: {summary_csv}")
