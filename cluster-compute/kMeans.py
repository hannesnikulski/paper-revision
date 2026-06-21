from scipy.spatial.distance import squareform
import time
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import numpy as np


# ============================================================
# CONFIG
# ============================================================
N_CLUSTERS = 4

for groupID in [0, 1, 2, 3, 4]:
    if groupID != 0:
        continue

    DIST_FILE = f"data/cluster/distances/dist_group_{groupID}_f16.dat"
    KEYS_FILE = f"data/cluster/distances/compact_group_{groupID}_keys.npy"

    # ============================================================
    # LOAD KEYS
    # ============================================================

    keys = np.load(KEYS_FILE, allow_pickle=True)
    n = len(keys)

    # print(f"n = {n}")

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

    # distances = np.asarray(distances, dtype=np.float64)
    D = squareform(distances)

    def format_bytes(n_bytes):
        units = ["B", "KB", "MB", "GB", "TB", "PB"]

        size = float(n_bytes)

        for unit in units:
            if size < 1024 or unit == units[-1]:
                return f"{size:.2f} {unit}"
            size /= 1024

    print("Matrix size:", format_bytes(D.nbytes))

    startTime = time.time()

    print("start embedding")

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=0
    )

    # Generate embedded observations with given distances
    X = mds.fit_transform(D)

    endTime = time.time()
    print("Embedding took:", endTime - startTime, "seconds")
    print("start KMeans")

    # Perform kMeans clustering
    kmeans = KMeans(n_clusters=2, random_state=0)
    labels = kmeans.fit_predict(X)

    # model = DBSCAN(metric="precomputed", eps=0.5, min_samples=5)
    # labels = model.fit_predict(D)

    endTime = time.time()
    print("Calculation took:", endTime - startTime, "seconds")

    np.save(f"data/cluster/distances/kMeans_labels_group_{groupID}.npy", labels)
