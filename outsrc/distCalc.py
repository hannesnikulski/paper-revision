import time
from scipy.spatial.distance import pdist, squareform

import pandas as pd
import numpy as np
from tqdm import tqdm


def dist(a, b):
    a = np.array(a)
    b = np.array(b)

    mask = ~np.isnan(a) & ~np.isnan(b)

    if mask.sum() == 0:
        return 100

    diff = a[mask] - b[mask]
    return np.sqrt(np.sum(diff**2) / mask.sum())


start = time.time()
for groupID in [0, 1, 2, 3, 4]:
    df = pd.read_parquet(f"compact_group_{groupID}.parquet")

    X = []
    X_keys = []
    zgrid = np.linspace(-1, 1, 50)
    for key, sub in tqdm(df.groupby(["Date", "Texp"])):
        sub = sub.dropna(subset=["z", "w"])
        z = sub["z"].to_numpy()
        w = sub["w"].to_numpy()

        z_max_abs = np.abs(z).max()

        if z_max_abs == 0:
            continue

        z_norm = z / z_max_abs
        w_norm = w / w.mean()

        w_interp = np.interp(zgrid, z_norm, w_norm, left=np.nan, right=np.nan)
        X.append(w_interp)
        X_keys.append(key)

    X = np.array(X)
    distances = pdist(X, metric=lambda u, v: dist(u, v))
    np.save(f"dist_group_{groupID}.npy", distances)

    current = time.time()
    print("time: ", current - start)
