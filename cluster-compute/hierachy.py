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

for groupID in [0, 1, 2, 3, 4]:
    DIST_FILE = f"data/cluster/distances/dist_group_{groupID}_f16.npyw"
    KEYS_FILE = f"data/cluster/distances/comapct_group_{groupID}_keys.npy"

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

    def format_bytes(n_bytes):
        units = ["B", "KB", "MB", "GB", "TB", "PB"]

        size = float(n_bytes)

        for unit in units:
            if size < 1024 or unit == units[-1]:
                return f"{size:.2f} {unit}"
            size /= 1024

    print("Matrix size:", format_bytes(distances.nbytes))

    # ============================================================
    # HIERARCHICAL CLUSTERING
    # ============================================================

    startTime = time.time()

    Z = linkage(distances, method="complete")
    labels = fcluster(Z, t=N_CLUSTERS, criterion="maxclust")

    endTime = time.time()
    print("Calculation took:", endTime - startTime, "seconds")

    np.save(f"data/cluster/distances/compact_hierachy_labels_group_{groupID}.npy", labels)
