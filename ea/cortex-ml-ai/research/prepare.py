"""Build features + candidate labels and cache them.

usage: python prepare.py <out.parquet> [tp] [act] [lock] [dist] [tight]
Every closed signal bar produces two candidates (long and short).  Each candidate
is simulated on the M1 path with the exact trade management of the EA.
"""
import sys, time
import numpy as np, pandas as pd
import engine as E
import features as F

out = sys.argv[1]
tp = float(sys.argv[2]) if len(sys.argv) > 2 else 13.5
act = float(sys.argv[3]) if len(sys.argv) > 3 else 7.0
lock = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
dist = float(sys.argv[5]) if len(sys.argv) > 5 else 4.0
tight = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0

t0 = time.time()
m1 = E.load_m1()
X = F.build(m1, tp=tp)
X = X.dropna()
print("features", X.shape, round(time.time() - t0, 1), "s")

tt = m1.stime.values.astype("datetime64[s]").astype(np.int64)
o, h, l, c = (m1[k].values.astype(np.float64) for k in ["open", "high", "low", "close"])
bar_end = (X.index + pd.Timedelta(F.SIGNAL_TF)).values.astype("datetime64[s]").astype(np.int64)
ent = np.searchsorted(tt, bar_end)
ok = (ent < len(tt)) & ((tt[np.minimum(ent, len(tt) - 1)] - bar_end) < 3600)
X, ent = X[ok], ent[ok]

# data holes: weekday gaps > 90 min that are not the weekend break -> trades spanning them are invalid
gap = np.diff(tt, prepend=tt[0]) / 60.0
wd = pd.DatetimeIndex(m1.stime.values).weekday.values
hole = (gap > 90) & ~((wd == 0) & (gap > 24 * 60)) & ~((wd == 6) & (gap > 24 * 60))
hole_cum = np.cumsum(hole)
import os
prm = E.mgmt(tp=tp, act=act, lock=lock, dist=dist, tight=tight, maxhold=0,
             spread=float(os.environ.get("SPREAD", 0.25)), comm=float(os.environ.get("COMM", 0.07)))
rows = []
for d in (1, -1):
    idx = ent.astype(np.int64)
    dirs = np.full(len(idx), d, dtype=np.int64)
    tpa = np.full(len(idx), tp)
    dsa = np.full(len(idx), dist)
    pnl, ex, rs, mfe, mae = E.label_all(idx, dirs, tt, o, h, l, c, prm, tpa, dsa)
    Y = F.mirror(X, d)
    Y["dir"] = d
    Y["ent"] = idx
    Y["ex"] = ex
    Y["pnl"] = pnl
    Y["reason"] = rs
    Y["mfe"] = mfe
    Y["mae"] = mae
    Y["hold"] = (tt[ex] - tt[idx]) / 60.0
    Y["valid"] = (hole_cum[ex] == hole_cum[idx]) & (m1.res.values[ex] <= 5)
    rows.append(Y)
D = pd.concat(rows).sort_index(kind="stable")
D.to_parquet(out)
print("candidates", len(D), round(time.time() - t0, 1), "s")
g = D.groupby([D.index.year, "dir"])
print(g.agg(n=("pnl", "size"), avg=("pnl", "mean"), wr=("pnl", lambda x: (x > 0).mean()),
            sl=("reason", lambda x: (x == 2).mean()), tp=("reason", lambda x: (x == 1).mean()),
            hold=("hold", "median")).round(3).to_string())
