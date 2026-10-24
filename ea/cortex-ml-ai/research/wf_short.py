"""Short-memory online learning: retrain every week on the last N days of resolved candidates.
Tests whether regime persistence can be harvested by an adaptive model."""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import engine as E
from rules import load

name = sys.argv[1]
win_days = int(sys.argv[2])
kind = sys.argv[3]               # lgb | logit
thr_list = [float(x) for x in sys.argv[4].split(",")]
D = load(name)
D = D[D.res <= 5].copy()
m1 = E.load_m1()
tt = m1.stime.values
D["exit_time"] = tt[D.ex.values]
D["y"] = (D.pnl > 0).astype(int)
hr = D.index.hour
D = D[~((hr == 23) | (hr == 0))]          # skip rollover hours (wide spreads)
META = ["ent", "ex", "pnl", "reason", "mfe", "mae", "hold", "close", "res", "exit_time", "y"]
feats = [k for k in D.columns if k not in META]
weeks = pd.date_range("2021-01-04", "2026-10-19", freq="W-MON")
preds = []
for a in weeks:
    b = a + pd.Timedelta(days=7)
    tr = D[(D.exit_time < a) & (D.index >= a - pd.Timedelta(days=win_days))]
    te = D[(D.index >= a) & (D.index < b)]
    if len(te) == 0 or len(tr) < 500:
        continue
    if kind == "lgb":
        mdl = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=7, min_child_samples=100,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=10, verbose=-1)
        mdl.fit(tr[feats], tr.y)
        p = mdl.predict_proba(te[feats])[:, 1]
    else:
        sc = StandardScaler().fit(tr[feats])
        mdl = LogisticRegression(C=0.05, max_iter=300)
        mdl.fit(np.clip(sc.transform(tr[feats]), -5, 5), tr.y)
        p = mdl.predict_proba(np.clip(sc.transform(te[feats]), -5, 5))[:, 1]
    base = tr.y.mean()
    preds.append(pd.DataFrame({"p": p, "edge": p - base}, index=te.index).assign(
        dir=te.dir.values, ent=te.ent.values, ex=te.ex.values, pnl=te.pnl.values, res=te.res.values))
P = pd.concat(preds).sort_values(["ent", "dir"])
P.to_parquet(f"/tmp/claude-0/-home-user-quantix-site-live/d63ac0eb-0794-5f04-8945-165b7e1ba43f/scratchpad/data/P_{name}_{win_days}_{kind}.parquet")
from sklearn.metrics import roc_auc_score
print(name, win_days, kind, " AUC by year:", {y: round(roc_auc_score(q.pnl > 0, q.p), 3) for y, q in P.groupby(P.index.year)})
for thr in thr_list:
    take = P.p.values >= thr
    done = E.portfolio(take, P.ent.values, P.ex.values, P.pnl.values, 1)
    T = P[done]
    yr = {y: (round(q.pnl.sum()), len(q)) for y, q in T.groupby(T.index.year)}
    last = T[T.index >= "2025-04-24"]
    print(f" thr {thr}", yr, "| L18m", E.stats(last.pnl.values, last.index))
