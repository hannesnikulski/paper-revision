from scipy.spatial.distance import squareform
import time
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.interpolate import interp1d

# ============================================================
# CONFIG
# ============================================================

N_CLUSTERS = 4

DIST_FILE = "matrix_group_0.dat"
KEYS_FILE = "matrix_group_0_keys.npy"
PARQUET_FILE = "group_0.parquet"

# ============================================================
# LOAD KEYS
# ============================================================

keys = np.load(KEYS_FILE, allow_pickle=True)
n = len(keys)

print(f"n = {n}")

# ============================================================
# LOAD CONDENSED DISTANCE MATRIX
# ============================================================

m = n * (n - 1) // 2

distances = np.memmap(
    DIST_FILE,
    mode="r",
    dtype=np.float16,
    shape=(m,)
)

distances = np.asarray(distances, dtype=np.float64)
D = squareform(distances)
print(D.nbytes)

s = time.time()

mds = MDS(
    n_components=2,
    dissimilarity="precomputed",
    random_state=0
)

X = mds.fit_transform(D)

kmeans = KMeans(n_clusters=2, random_state=0)
labels = kmeans.fit_predict(X)
e = time.time()
print(e - s)

quit()

# ============================================================
# HIERARCHICAL CLUSTERING
# ============================================================

Z = linkage(distances, method="complete")
labels = fcluster(Z, t=N_CLUSTERS, criterion="maxclust")

# ============================================================
# BUILD CLUSTER TABLE
# ============================================================

cluster_df = pd.DataFrame({
    "Date": [k[0] for k in keys],
    "Texp": [k[1] for k in keys],
    "cluster": labels
})

cluster_df["Date"] = pd.to_datetime(cluster_df["Date"])

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(PARQUET_FILE)
df["Date"] = pd.to_datetime(df["Date"])

# ============================================================
# MERGE CLUSTERS
# ============================================================

df = df.merge(cluster_df, on=["Date", "Texp"], how="inner")

print("Merged rows:", len(df))

# ============================================================
# PROCESS CLUSTERS
# ============================================================

cluster_results = {}

for cluster_id in sorted(df["cluster"].unique()):

    members = df[df["cluster"] == cluster_id][["Date", "Texp"]].drop_duplicates()

    curves = []

    for _, row in tqdm(members.iterrows()):

        curve = df[
            (df["Date"] == row["Date"]) &
            (df["Texp"] == row["Texp"])
        ][["z", "w"]].dropna()

        curve = curve.sort_values("z")

        if len(curve) < 2:
            continue

        curves.append(curve)

    if len(curves) == 0:
        continue

    # ========================================================
    # UNION GRID (allow NaNs outside each curve domain)
    # ========================================================

    z_min = min(c["z"].min() for c in curves)
    z_max = max(c["z"].max() for c in curves)

    z_grid = np.linspace(z_min, z_max, 100)

    interp_curves = []

    for curve in curves:

        f = interp1d(
            curve["z"].values,
            curve["w"].values,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan
        )

        interp_curves.append(f(z_grid))

    interp_curves = np.vstack(interp_curves)

    mean_w = np.nanmean(interp_curves, axis=0)
    var_w = np.nanvar(interp_curves, axis=0)
    std_w = np.sqrt(var_w)

    coverage = np.sum(~np.isnan(interp_curves), axis=0)

    cluster_results[cluster_id] = {
        "z": z_grid,
        "mean": mean_w,
        "std": std_w,
        "var": var_w,
        "coverage": coverage,
        "n_curves": len(curves),
    }

# ============================================================
# PLOT MEAN ± STD
# ============================================================

plt.figure(figsize=(12, 7))

for cid, res in cluster_results.items():

    plt.plot(res["z"], res["mean"], label=f"Cluster {cid} (n={res['n_curves']})")

    plt.fill_between(
        res["z"],
        res["mean"] - res["std"],
        res["mean"] + res["std"],
        alpha=0.2
    )

plt.xlabel("z")
plt.ylabel("w")
plt.title("Cluster Mean Curves ± 1 Std (Interpolated)")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# PLOT COVERAGE
# ============================================================

plt.figure(figsize=(12, 4))

for cid, res in cluster_results.items():
    plt.plot(res["z"], res["coverage"], label=f"Cluster {cid}")

plt.xlabel("z")
plt.ylabel("# contributing curves")
plt.title("Interpolation Coverage per Cluster")
plt.legend()
plt.grid(True)
plt.show()
