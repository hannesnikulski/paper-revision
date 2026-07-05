import multiprocessing as mp
import numpy as np
import pandas as pd

from itertools import combinations, islice
from tqdm import tqdm

Array = np.ndarray
MAX_DISTANCE = 100
minNumberOfSlices = 10
numWorkers = mp.cpu_count() - 1
batchSize = 500


################################################################################
# Paths
################################################################################
def getDataPath(groupID: int) -> str:
    return f"data/intermediate/group_{groupID}_slices.npy"

def getDistancePath(groupID: int) -> str:
    return f"data/intermediate/group_{groupID}_distances.npy"

def triIndex(i: int, j: int, n: int) -> int:
    return n * i - i * (i + 1) // 2 + (j - i - 1)


################################################################################
# Distance function
################################################################################
def distance(x: Array, y: Array) -> float:
    mask = ~np.isnan(x) & ~ np.isnan(y)

    if mask.sum() == 0:
        return MAX_DISTANCE
    
    diff = x[mask] - y[mask]
    return np.sqrt(np.sum(diff ** 2)) / mask.sum()


################################################################################
# Worker
################################################################################
def worker(
    taskQueue:mp.Queue,
    resultQueue: mp.Queue,
    groupID: int,
    n: int,
    flatSize: int
):
    slices = np.load(getDataPath(groupID))
    distances = np.memmap(getDistancePath(groupID), dtype=np.float16, mode="r+", shape=(flatSize,))

    while True:
        batch = taskQueue.get()
        if batch is None:
            del distances
            break

        for i, j in batch:
            distances[triIndex(i, j, n)] = distance(slices[i], slices[j])

        distances.flush()
        resultQueue.put(len(batch))


def iterBatches(n: int, batchSize: int):
    iterator = combinations(range(n), 2)
    while True:
        batch = list(islice(iterator, batchSize))
        if not batch:
            break

        yield batch


################################################################################
# Main Process
################################################################################
if __name__ == "__main__":
    df = pd.read_parquet("data/ivdata.parquet")
    
    groupIDs = [0, 1, 2, 3, 4]

    zgrid = np.linspace(-5, 5, 100)
    for idx, (groupLabel, group) in enumerate(df.groupby("Group")):
        if idx not in groupIDs:
            continue

        sliceMatrix = []
        sliceKeys = []

        for key, subgroup in tqdm(group.groupby(["Date", "Texp"])):
            subgroup = subgroup.dropna(subset=["z", "w"])

            z = subgroup["z"]
            w = subgroup["w"]
            
            # zMaxValue = np.abs(z).max()
            # if zMaxValue == 0:
            #     continue

            # zNormalized = z / zMaxValue
            wNormalized = w / w.mean()

            wInterp = np.interp(zgrid, z, wNormalized, left=np.nan, right=np.nan)
            
            sliceMatrix.append(wInterp)
            sliceKeys.append(key)

        n = len(sliceMatrix)
        if n < minNumberOfSlices:
            continue

        sliceMatrix = np.array(sliceMatrix)
        sliceKeys = np.array(sliceKeys)

        np.save(getDataPath(idx), sliceMatrix)
        np.save(f"data/intermediate/group_{idx}_keys.npy", sliceKeys)

        flatSize = n * (n - 1) // 2

        matrix = np.memmap(getDistancePath(idx), dtype=np.float16, mode="w+", shape=(flatSize,))
        del matrix

        taskQueue = mp.Queue(maxsize=numWorkers * 4)
        resultQueue = mp.Queue()

        procs = [
            mp.Process(
                target=worker,
                args=(taskQueue, resultQueue, idx, n, flatSize)
            )
            for _ in range(numWorkers)
        ]

        for p in procs:
            p.start()

        with tqdm(total=flatSize, desc=f"Group {idx} ({n} slices, {flatSize} pairs)") as pbar:
            completedBatches = 0
            totalBatches = 0

            for batch in iterBatches(n, batchSize):
                taskQueue.put(batch)
                totalBatches += 1

                while not resultQueue.empty():
                    pbar.update(resultQueue.get_nowait())
                    completedBatches += 1

            for _ in range(numWorkers):
                taskQueue.put(None)

            while completedBatches < totalBatches:
                pbar.update(resultQueue.get())
                completedBatches += 1

        for p in procs:
            p.join()
