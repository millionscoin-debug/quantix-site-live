"""Walk-forward logistic meta-model on the momentum engine (the model the EA runs).

args: name gate win_days C retrain(M|W) margins hours(h0-h1) [featureset]
"""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
import engine as E
from rules import load

name, gate, win_days, C, freq = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
margins = [float(x) for x in sys.argv[6].split(",")]
h0, h1 = (float(x) for x in sys.argv[7].split("-"))
fset = sys.argv[8] if len(sys.argv) > 8 else "all"
half_life = float(sys.argv[9]) if len(sys.argv) > 9 else 0.0
D = load(name)
hr = D.index.hour + D.index.minute / 60
D = D[(D.d_today > gate) & (hr >= h0) & (hr < h1) & (D.res <= 5) & D.valid].copy()
m1 = E.load_m1()
D["exit_time"] = m1.stime.values[D.ex.values]
D["y"] = (D.pnl > 0).astype(int)
META = ["ent", "ex", "pnl", "reason", "mfe", "mae", "hold", "close", "res", "exit_time", "y", "valid"]
feats = [k for k in D.columns if k not in META]
if fset != "all":
    import json
    feats = json.loads(open(fset).read())
X = D[feats].values.astype(float)
periods = pd.date_range("2021-01-01", "2026-10-24", freq="MS" if freq == "M" else "W-MON")
P = np.full(len(D), np.nan)
B = np.full(len(D), np.nan)
idx_time = D.index.values
et = D.exit_time.values
for i, a in enumerate(periods):
    b = periods[i + 1] if i + 1 < len(periods) else pd.Timestamp("2026-12-31")
    a64, b64 = np.datetime64(a), np.datetime64(b)
    tr = (et < a64) & (idx_time >= a64 - np.timedelta64(win_days, "D"))
    te = (idx_time >= a64) & (idx_time < b64)
    if te.sum() == 0 or tr.sum() < 200:
        continue
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Z = np.clip((X[tr] - mu) / sd, -4, 4)
    w = None
    if half_life > 0:
        age = (a64 - idx_time[tr]) / np.timedelta64(1, "D")
        w = 0.5 ** (age / half_life)
        w = w / w.mean()
    lr = LogisticRegression(C=C, max_iter=500).fit(Z, D.y.values[tr], sample_weight=w)
    P[te] = lr.predict_proba(np.clip((X[te] - mu) / sd, -4, 4))[:, 1]
    B[te] = D.y.values[tr].mean()
D["p"], D["base"] = P, B
D = D.dropna(subset=["p"])
tag = f"g{gate}_w{win_days}_C{C}_{freq}_h{h0:g}-{h1:g}_{fset.split('/')[-1].replace('.json', '')}_hl{half_life:g}"
D[["p", "base", "dir", "ent", "ex", "pnl", "reason", "hold", "d_today"]].to_parquet(
    f"/tmp/claude-0/-home-user-quantix-site-live/d63ac0eb-0794-5f04-8945-165b7e1ba43f/scratchpad/data/LR_{name}_{tag}.parquet")
for mg in margins:
    take = (D.p > D.base + mg).values
    done = E.portfolio(take, D.ent.values, D.ex.values, D.pnl.values, 1)
    T = D[done]
    by = {y: round(q.pnl.sum()) for y, q in T.groupby(T.index.year)}
    last = T[T.index >= "2025-04-24"]
    st = E.stats(last.pnl.values, last.index)
    print(f"{tag} m{mg}: {by} n={len(T)} | L18 net {st['net']} n {st['n']} pf {st['pf']} wr {st['wr']} dd {st['maxdd']} r2 {st['r2']} mo+ {st['mo_pos']}")
