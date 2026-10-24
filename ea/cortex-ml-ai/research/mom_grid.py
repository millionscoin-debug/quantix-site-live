"""Robustness grid for the strong-day momentum engine: re-simulate the management on the
subset of candidates with d_today > 0.6 for many (TP, trailing, k, hours, max_pos) combos."""
import sys, itertools
import numpy as np, pandas as pd
import engine as E
from rules import load

D = load("cand_t5_4")
hr = D.index.hour + D.index.minute / 60
D = D[(D.d_today > 0.6) & ~((hr >= 23) | (hr < 1.5)) & (D.res <= 5)].copy()
m1 = E.load_m1()
tt = m1.stime.values.astype("datetime64[s]").astype(np.int64)
o, h, l, c = (m1[k].values.astype(np.float64) for k in ["open", "high", "low", "close"])
idx = D.ent.values.astype(np.int64)
dirs = D.dir.values.astype(np.int64)
hrs = D.index.hour.values + D.index.minute.values / 60
yrs = D.index.year.values
rows = []
cfgs = list(itertools.product([12.0, 13.5, 15.0], [(5, 4, 3), (5, 3.5, 3), (6, 4, 3), (6, 5, 3), (7, 4, 3), (7, 5, 3), (8, 5, 4), (99, 3, 4)], [1.0, 0.6]))
for tp, (act, lock, dist), tight in cfgs:
    if act == 99 and tight != 1.0:
        continue
    prm = E.mgmt(tp=tp, act=act, lock=lock, dist=dist, tight=tight)
    pnl, ex, rs, _, _ = E.label_all(idx, dirs, tt, o, h, l, c, prm, np.full(len(idx), tp), np.full(len(idx), float(dist)))
    valid = D.valid.values
    for k in (0.8, 0.9, 1.0, 1.1, 1.2):
        for h0, h1 in ((1.5, 23), (8, 23), (12, 23), (1.5, 20)):
            for mp in (1, 2):
                take = (D.d_today.values > k) & (hrs >= h0) & (hrs < h1) & valid
                done = E.portfolio(take, idx, ex, pnl, mp)
                p, y = pnl[done], yrs[done]
                by = {yy: p[y == yy].sum() for yy in range(2020, 2027)}
                tl = D.index[done]
                last = p[tl >= "2025-04-24"]
                st = E.stats(last, tl[tl >= "2025-04-24"])
                rows.append({"tp": tp, "act": act, "lock": lock, "dist": dist, "tight": tight, "k": k, "h0": h0, "h1": h1,
                             "mp": mp, "pos_yrs": sum(v > 0 for v in by.values()), "min_yr": min(by.values()),
                             "tot": p.sum(), "n": len(p), **{str(a): round(b) for a, b in by.items()},
                             "L18": st["net"], "L18pf": st["pf"], "L18dd": st["maxdd"], "L18r2": st["r2"], "L18wr": st["wr"]})
R = pd.DataFrame(rows)
R.to_parquet("/tmp/claude-0/-home-user-quantix-site-live/d63ac0eb-0794-5f04-8945-165b7e1ba43f/scratchpad/data/mom_grid.parquet")
pd.set_option("display.width", 260)
print(R.sort_values(["pos_yrs", "min_yr"], ascending=False).head(30).to_string())
print(R.sort_values("L18", ascending=False).head(15).to_string())
