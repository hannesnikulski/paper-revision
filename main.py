import pandas as pd
import pyreadr

res = pyreadr.read_r("data/spxOptionMetricsIvols.rData")
print(res)
