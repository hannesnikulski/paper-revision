import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import combinations, islice
from tqdm import tqdm

min_points_per_slice = 20
num_grid_points = 20
batch_size = 500
num_workers = mp.cpu_count()


def tril_index(i: int, j: int, n: int) -> int:
    return n * i - i * (i + 1) // 2 + (j - i - 1)


def preprocess(z: np.ndarray, w: np.ndarray):
    return z, w


def ext_dist(z_i, w_i, z_j, w_j) -> float:
    z_min = max(z_i.min(), z_j.min())
    z_max = min(z_i.max(), z_j.max())
    if z_max <= z_min:
        return 0.0
    z_grid = np.linspace(z_min, z_max, num_grid_points)
    w_i_interp = np.interp(z_grid, z_i, w_i)
    w_j_interp = np.interp(z_grid, z_j, w_j)
    return float(np.linalg.norm(w_i_interp - w_j_interp))


def worker(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    slices_path: str,        # workers load slices from disk instead of pickle
    matrix_path: str,
    n: int,
    flat_size: int,
) -> None:
    # Load slices from disk — no pickle transfer
    slices: list[tuple[np.ndarray, np.ndarray]] = np.load(slices_path, allow_pickle=True).tolist()
    matrix = np.memmap(matrix_path, dtype=np.float16, mode="r+", shape=(flat_size,))

    while True:
        batch = task_queue.get()
        if batch is None:
            del matrix
            break

        for i, j in batch:
            z_i, w_i = preprocess(*slices[i])
            z_j, w_j = preprocess(*slices[j])
            matrix[tril_index(i, j, n)] = ext_dist(z_i, w_i, z_j, w_j)

        matrix.flush()
        result_queue.put(len(batch))


def iter_batches(n: int, batch_size: int):
    """Yield batches of pairs without materialising all pairs at once."""
    it = combinations(range(n), 2)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch


if __name__ == "__main__":
    groups = [0, 3, 4]
    dfs = []
    for g in groups:
        _df = pd.read_parquet(f"group_{g}.parquet")
        dfs.append((g, _df))

    for mi, (maturityLabel, maturityGroup) in enumerate(dfs):
        slices: list[tuple[np.ndarray, np.ndarray]] = []
        slice_keys: list[tuple] = []

        for (date, texp), group in maturityGroup.groupby(["Date", "Texp"]):
            group = group.dropna(subset=["z", "w"])
            if group.empty or len(group) < min_points_per_slice:
                continue
            z = group["z"].to_numpy(dtype=np.float32)
            w = group["w"].to_numpy(dtype=np.float32)
            if np.abs(z).max() == 0:
                continue
            order = np.argsort(z)
            slices.append((z[order], w[order]))
            slice_keys.append((date, texp))

        slices = slices[:2000]
        slice_keys = slice_keys[:2000]

        n = len(slices)
        if n < 2:
            continue

        # Save slices to disk so workers load them independently (no pickle transfer)
        slices_path = f"matrix_group_{mi}_slices.npy"
        np.save(slices_path, np.array(slices, dtype=object))

        flat_size = n * (n - 1) // 2
        matrix_path = f"matrix_group_{mi}.dat"

        matrix = np.memmap(matrix_path, dtype=np.float16, mode="w+", shape=(flat_size,))
        del matrix

        total_pairs = n * (n - 1) // 2
        task_queue: mp.Queue = mp.Queue(maxsize=num_workers * 4)  # bounded — don't pre-fill RAM
        result_queue: mp.Queue = mp.Queue()

        processes = [
            mp.Process(
                target=worker,
                args=(task_queue, result_queue, slices_path, matrix_path, n, flat_size),
            )
            for _ in range(num_workers)
        ]
        for p in processes:
            p.start()

        # Feed batches lazily from the main process
        with tqdm(total=total_pairs, desc=f"Group {mi} ({n} slices, {total_pairs} pairs)") as pbar:
            completed_batches = 0
            total_batches = 0

            for batch in iter_batches(n, batch_size):
                task_queue.put(batch)   # blocks if queue is full (backpressure)
                total_batches += 1

                # Drain any results that are ready without blocking
                while not result_queue.empty():
                    pbar.update(result_queue.get_nowait())
                    completed_batches += 1

            # Poison pills
            for _ in range(num_workers):
                task_queue.put(None)

            # Drain remaining results
            while completed_batches < total_batches:
                pbar.update(result_queue.get())
                completed_batches += 1

        for p in processes:
            p.join()

        # Clean up slices file
        import os
        os.remove(slices_path)

        print(f"Group {mi}: {total_pairs} pairs computed")
        np.save(f"matrix_group_{mi}_keys.npy", np.array(slice_keys, dtype=object))

        # Free the group dataframe
        del slices, slice_keys
        dfs[mi] = (maturityLabel, None)  # release the dataframe reference
