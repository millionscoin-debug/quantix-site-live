"""Univariate stability scan: mean P/L of candidates per feature decile, per year.
A feature is interesting when an extreme decile has the same sign of edge in most years."""
import sys
import numpy as np, pandas as pd

D = pd.read_parquet(sys.argv[1])
D = D[D.res <= 5]
META = ["ent", "ex", "pnl", "reason", "mfe", "mae", "hold", "close", "res", "dir"]
feats = [k for k in D.columns if k not in META]
D["yr"] = D.index.year
rows = []
for f in feats:
    try:
        q = D.groupby("yr")[f].transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False))
    except Exception:
        continue
    for dec in (0, 9):
        sub = D[q == dec]
        for side in (1, -1, 0):
            s = sub if side == 0 else sub[sub.dir == side]
            by = s.groupby("yr").pnl.mean()
            rows.append((f, dec, side, by.mean(), (by > 0).sum(), len(by), by.min(), *by.round(2).values))
R = pd.DataFrame(rows, columns=["feat", "dec", "side", "mean", "pos", "yrs", "min"] + [str(y) for y in sorted(D.yr.unique())])
R = R.sort_values("min", ascending=False)
pd.set_option("display.width", 250)
print(R.head(40).round(2).to_string())
