"""Slow-trend alignment: candidates taken only in the direction of a daily trend filter."""
import sys
import numpy as np, pandas as pd
import engine as E, features as F

D = pd.read_parquet(sys.argv[1])
D = D[D.res <= 5].copy()
m1 = E.load_m1()
d1 = F.resample(m1, "1D")
c = d1.close
defs = {}
for fast, slow in [(10, 30), (20, 50), (50, 200)]:
    defs[f"ema{fast}>{slow}"] = (F.ema(c.values, fast) > F.ema(c.values, slow)).astype(int) * 2 - 1
for n in (10, 20, 50):
    e = F.ema(c.values, n)
    defs[f"c>ema{n}"] = np.sign(c.values - e)
    defs[f"slope_ema{n}"] = np.sign(e - np.roll(e, 3))
for n in (5, 10, 20, 40):
    defs[f"mom{n}"] = np.sign(c.values - np.roll(c.values, n))
T = pd.DataFrame(defs, index=d1.index).shift(1)          # yesterday's daily value, known today
day = D.index.normalize()
D["yr"] = D.index.year
rows = []
for k in T.columns:
    sgn = T[k].reindex(day).values
    m = sgn == D.dir.values
    s = D[m]
    by = s.groupby("yr").pnl.mean()
    rows.append({"filter": k, "pos": int((by > 0).sum()), "mean": s.pnl.mean(), **by.round(2).to_dict()})
pd.set_option("display.width", 220)
print(pd.DataFrame(rows).round(2).to_string())
