import time

import kmedoids
import numpy as np

from scipy.spatial.distance import squareform


# ============================================================
# CONFIG
# ============================================================

N_CLUSTERS = 4

for groupID in [0, 1, 2, 3, 4]:
    DIST_FILE = f"data/cluster/distances/dist_group_{groupID}_f16.npy"
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
    # K-MEDOIDS CLUSTERING (FasterPAM)
    # ============================================================

    startTime = time.time()

    # Convert condensed distance vector -> full distance matrix
    D = squareform(distances)

    # Run K-Medoids
    result = kmedoids.fasterpam(
        D,
        medoids=N_CLUSTERS,
        random_state=42
    )

    # Labels are 0-based cluster assignments
    labels = result.labels

    endTime = time.time()
    print("Calculation took:", endTime - startTime, "seconds")

    # Save labels
    np.save(
        f"data/cluster/distances/cmpt_kmedoids_labels_group_{groupID}.npy",
        labels
    )

    # Save medoids (optional but useful)
    np.save(
        f"data/cluster/distances/cmpt_kmedoids_medoids_group_{groupID}.npy",
        result.medoids
    )
