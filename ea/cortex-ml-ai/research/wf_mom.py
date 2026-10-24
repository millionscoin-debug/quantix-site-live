"""ML meta-filter on top of the strong-day momentum engine (walk-forward, purged, monthly retrain).
Compares the plain rule (d_today > k) with model-filtered versions."""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import engine as E
from rules import load

name = sys.argv[1] if len(sys.argv) > 1 else "cand_t5_4"
kmin = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
win_years = float(sys.argv[3]) if len(sys.argv) > 3 else 3
D = load(name)
hr = D.index.hour + D.index.minute / 60
D = D[(D.d_today > kmin) & ~((hr >= 23) | (hr < 1.5)) & (D.res <= 5) & D.valid].copy()
m1 = E.load_m1()
D["exit_time"] = m1.stime.values[D.ex.values]
D["y"] = (D.pnl > 0).astype(int)
META = ["ent", "ex", "pnl", "reason", "mfe", "mae", "hold", "close", "res", "exit_time", "y", "valid"]
feats = [k for k in D.columns if k not in META]
out = []
for mo in pd.period_range("2021-01", "2026-10", freq="M"):
    a, b = mo.start_time, mo.end_time
    tr = D[(D.exit_time < a) & (D.index >= a - pd.Timedelta(days=int(365 * win_years)))]
    te = D[(D.index >= a) & (D.index <= b)]
    if len(te) == 0:
        continue
    g = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=60,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=10, verbose=-1)
    g.fit(tr[feats], tr.y)
    sc = StandardScaler().fit(tr[feats])
    lr = LogisticRegression(C=0.05, max_iter=500).fit(np.clip(sc.transform(tr[feats]), -5, 5), tr.y)
    out.append(te.assign(p_lgb=g.predict_proba(te[feats])[:, 1],
                         p_lr=lr.predict_proba(np.clip(sc.transform(te[feats]), -5, 5))[:, 1],
                         base=tr.y.mean()))
P = pd.concat(out).sort_values(["ent", "dir"])
P.to_parquet(f"/tmp/claude-0/-home-user-quantix-site-live/d63ac0eb-0794-5f04-8945-165b7e1ba43f/scratchpad/data/Pmom_{name}_{kmin}_{win_years}.parquet")
from sklearn.metrics import roc_auc_score
for col in ("p_lgb", "p_lr"):
    print(col, "AUC", {y: round(roc_auc_score(q.y, q[col]), 3) for y, q in P.groupby(P.index.year)})


def show(mask, label):
    done = E.portfolio(np.asarray(mask), P.ent.values, P.ex.values, P.pnl.values, 1)
    T = P[done]
    by = {y: round(q.pnl.sum()) for y, q in T.groupby(T.index.year)}
    last = T[T.index >= "2025-04-24"]
    st = E.stats(last.pnl.values, last.index)
    print(f"{label:28s} {by} n={len(T)} | L18 net {st['net']} pf {st['pf']} wr {st['wr']} dd {st['maxdd']} r2 {st['r2']}")


for k in (0.8, 1.0):
    show(P.d_today > k, f"rule d_today>{k}")
for col in ("p_lgb", "p_lr"):
    for q in (0.0, 0.02, 0.04, 0.06):
        show(P[col] > P.base + q, f"{col} > base+{q}")
    for k in (0.8,):
        for q in (0.0, 0.03):
            show((P.d_today > k) & (P[col] > P.base + q), f"rule{k} & {col}>base+{q}")
