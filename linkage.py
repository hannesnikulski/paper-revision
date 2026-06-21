import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster

# -------------------------
# Load clustering results
# -------------------------
n_clusters = 4

keys = np.load("matrix_group_0_keys.npy", allow_pickle=True)

# Reconstruct square matrix
n = len(keys)

memory_array = np.memmap(
    "matrix_group_0.dat",
    mode="r",
    dtype=np.float16,
    shape=(n, n)
)

Z = linkage(memory_array, method="complete", metric="precomputed")
labels = fcluster(Z, t=n_clusters, criterion="maxclust")

df = pd.read_parquet("group_0.parquet")

# -------------------------
# Build key -> cluster table
# -------------------------

# Example for keys = [(date, texp), ...]
cluster_df = pd.DataFrame(
    {
        "Date": [k[0] for k in keys],
        "Texp": [k[1] for k in keys],
        "cluster": labels,
    }
)

# ensure matching dtypes
cluster_df["Date"] = pd.to_datetime(cluster_df["Date"])
df["Date"] = pd.to_datetime(df["Date"])

# -------------------------
# Attach cluster labels
# -------------------------

df_clustered = df.merge(
    cluster_df,
    on=["Date", "Texp"],
    how="inner"
)

print(df_clustered.head())

# -------------------------
# Mean / variance by cluster
# -------------------------

stats = (
    df_clustered
    .groupby("cluster")
    .agg(
        z_mean=("z", "mean"),
        z_var=("z", "var"),
        w_mean=("w", "mean"),
        w_var=("w", "var"),
        n=("z", "size")
    )
    .reset_index()
)

print(stats)

# -------------------------
# Plot means
# -------------------------

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(stats))
width = 0.35

ax.bar(
    x - width / 2,
    stats["z_mean"],
    width,
    yerr=np.sqrt(stats["z_var"]),
    label="z"
)

ax.bar(
    x + width / 2,
    stats["w_mean"],
    width,
    yerr=np.sqrt(stats["w_var"]),
    label="w"
)

ax.set_xticks(x)
ax.set_xticklabels(stats["cluster"])
ax.set_xlabel("Cluster")
ax.set_ylabel("Mean")
ax.set_title("Cluster Mean ± Std")
ax.legend()

plt.tight_layout()
plt.show()
