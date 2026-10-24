"""Reference simulation of the final Cortex ML AI logic (what the EA does), used for the report.

Pipeline: candidate table (features + exact trade labels) -> monthly walk-forward logistic model
(3-year rolling window, purged) -> two-tier decision -> one-position execution -> stats/charts.

usage: python reference_sim.py <candidate table name> <out dir>
"""
import sys, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
import engine as E
from rules import load
from exec_opts import execute

CFG = dict(min_k=0.6, core_k=1.0, veto=0.03, margin=0.04, hour_from=8, hour_to=23, train_days=1095,
           C=0.05, max_pos=1, min_gap_min=0)
name, out = sys.argv[1], sys.argv[2]
D = load(name)
hr = D.index.hour + D.index.minute / 60
D = D[(D.d_today > CFG["min_k"]) & (hr >= CFG["hour_from"]) & (hr < CFG["hour_to"]) & (D.res <= 5) & D.valid].copy()
m1 = E.load_m1()
D["exit_time"] = m1.stime.values[D.ex.values]
D["y"] = (D.pnl > 0).astype(int)
META = ["ent", "ex", "pnl", "reason", "mfe", "mae", "hold", "close", "res", "exit_time", "y", "valid"]
feats = [k for k in D.columns if k not in META]
X = D[feats].values.astype(float)
it, et = D.index.values, D.exit_time.values
P, B = np.full(len(D), np.nan), np.full(len(D), np.nan)
months = pd.date_range("2021-01-01", "2026-11-01", freq="MS")
for a, b in zip(months[:-1], months[1:]):
    a64, b64 = np.datetime64(a), np.datetime64(b)
    tr = (et < a64) & (it >= a64 - np.timedelta64(CFG["train_days"], "D"))
    te = (it >= a64) & (it < b64)
    if te.sum() == 0 or tr.sum() < 300:
        continue
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    lr = LogisticRegression(C=CFG["C"], max_iter=500).fit(np.clip((X[tr] - mu) / sd, -4, 4), D.y.values[tr])
    P[te] = lr.predict_proba(np.clip((X[te] - mu) / sd, -4, 4))[:, 1]
    B[te] = D.y.values[tr].mean()
D["p"], D["base"] = P, B
D = D.dropna(subset=["p"]).sort_values(["ent", "dir"])
e = D.p - D.base
take = ((D.d_today >= CFG["core_k"]) & (e > -CFG["veto"])) | ((D.d_today < CFG["core_k"]) & (e > CFG["margin"]))
day = D.index.normalize().values.astype("datetime64[D]").astype(np.int64)
done = execute(take.values, D.ent.values, D.ex.values, D.pnl.values, D.reason.values.astype(np.int64), day,
               CFG["max_pos"], 99, 0, CFG["min_gap_min"])
T = D[done].copy()
T["exit_time"] = pd.to_datetime(T.exit_time)
T["tier"] = np.where(T.d_today >= CFG["core_k"], "core", "ml")
cols = ["dir", "tier", "d_today", "p", "base", "pnl", "reason", "hold", "exit_time"]
T[cols].to_csv(f"{out}/trades.csv")


def summary(q, label):
    st = E.stats(q.pnl.values, q.index)
    rs = q.reason.value_counts().to_dict()
    st.update({"label": label, "tp_exits": int(rs.get(1, 0)), "sl_exits": int(rs.get(2, 0)), "trail_exits": int(rs.get(3, 0)),
               "avg_win": round(q.pnl[q.pnl > 0].mean(), 2), "avg_loss": round(q.pnl[q.pnl < 0].mean(), 2),
               "median_hold_min": round(q.hold.median(), 0)})
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in st.items()}


res = {"config": CFG, "yearly": {}, "periods": []}
for y, q in T.groupby(T.index.year):
    res["yearly"][int(y)] = summary(q, str(y))
res["periods"].append(summary(T, "2021-01 .. 2026-10 (walk-forward OOS)"))
res["periods"].append(summary(T[T.index >= "2025-04-24"], "last 18 months 2025-04-24 .. 2026-10-23"))
mo = T.pnl.resample("ME").sum()
res["monthly"] = {str(k.date())[:7]: round(float(v), 1) for k, v in mo.items()}
json.dump(res, open(f"{out}/summary.json", "w"), indent=1)
print(json.dumps(res["periods"], indent=1))
print({k: (v["net"], v["n"], v["pf"]) for k, v in res["yearly"].items()})
