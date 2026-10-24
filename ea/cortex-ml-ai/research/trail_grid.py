"""How does the trailing configuration change the payoff of unconditional entries?
(average P/L per candidate per period; a martingale market would give -costs for every row)"""
import sys, itertools
import numpy as np, pandas as pd
import engine as E

D = pd.read_parquet(sys.argv[1])
D = D[D.res <= 5]
m1 = E.load_m1()
tt = m1.time.values.astype("datetime64[s]").astype(np.int64)
o, h, l, c = (m1[k].values.astype(np.float64) for k in ["open", "high", "low", "close"])
sub = D[D.index >= "2024-01-01"]
idx = sub.ent.values.astype(np.int64)
dirs = sub.dir.values.astype(np.int64)
per = np.where(sub.index < "2025-01-01", "2024", np.where(sub.index < "2026-01-01", "2025", "2026"))
rows = []
for tp, act, lock, dist, tight in itertools.product([12, 13.5, 15], [5, 7, 9, 11, 99], [3, 4, 6], [3, 4, 6, 8], [1.0, 0.6]):
    if act < 99 and (lock >= act or act > tp):
        continue
    if act == 99 and (lock != 3 or dist != 4 or tight != 1.0):
        continue
    prm = E.mgmt(tp=tp, act=act, lock=lock, dist=dist, tight=tight)
    pnl, ex, rs, _, _ = E.label_all(idx, dirs, tt, o, h, l, c, prm, np.full(len(idx), float(tp)), np.full(len(idx), float(dist)))
    r = {"tp": tp, "act": act, "lock": lock, "dist": dist, "tight": tight}
    for p in ("2024", "2025", "2026"):
        m = per == p
        r[p] = pnl[m].mean()
        r[p + "wr"] = (pnl[m] > 0).mean()
    rows.append(r)
R = pd.DataFrame(rows)
R["avg"] = R[["2024", "2025", "2026"]].mean(axis=1)
pd.set_option("display.width", 200)
print(R.sort_values("avg", ascending=False).round(3).head(25).to_string())
print(R[R.act == 99].round(3).to_string())
