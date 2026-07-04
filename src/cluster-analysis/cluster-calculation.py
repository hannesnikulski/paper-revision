import time

from pathlib import Path

import kmedoids
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# ============================================================
# Paths / Config
# ============================================================

projectPath = Path(__file__).parents[2]
datadir = projectPath / "data" / "z_unnorm"

N_CLUSTERS = 4

# ============================================================
# Clustering
# ============================================================
def format_bytes(n_bytes):
    units = ["B", "KB", "MB", "GB", "TB", "PB"]

    size = float(n_bytes)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
        
def hierarchyClustering(distances: np.ndarray, groupID: int):
    Z = linkage(distances, method="complete")
    labels = fcluster(Z, t=N_CLUSTERS, criterion="maxclust")

    # labels start at 0
    labels -= 1
    
    np.save(datadir / f"group_{groupID}_hierarchy_labels.npy", labels)

def kmedoidsClustering(D: np.ndarray, groupID: int):
    result = kmedoids.fasterpam(
        D,
        medoids=N_CLUSTERS,
    )
    labels = result.labels
    np.save(datadir / f"group_{groupID}_kmedoid_labels.npy", labels)

df = pd.read_parquet(projectPath / "data" / "ivdata.parquet")

for groupID, (groupLabel, group) in enumerate(df.groupby("Group")):
    keys = np.load(datadir / f"group_{groupID}_keys.npy", allow_pickle=True)
    n = len(keys)

    print(f"Group {groupID} | n = {n}")

    m = n * (n - 1) // 2

    distances = np.memmap(
    datadir / f"group_{groupID}_distances.npy",
    mode="r",
    dtype=np.float16,
    shape=(m,)
)
    distances = np.asarray(distances, dtype=np.float64)
    
    startTime = time.time()
    hierarchyClustering(distances, groupID)
    endTime = time.time()

    print(f"Clustering took: {endTime - startTime} seconds")
    
    startTime = time.time()
    D = squareform(distances)
    print(f"Matrix Size: {format_bytes(D.nbytes)}")
    kmedoidsClustering(D, groupID)
    endTime = time.time()
    
    print(f"Clustering took: {endTime - startTime} seconds")

    


