"""Reference simulation of Cortex ML AI v1.1 (two models + activity governor + smooth recovery).

Inputs: cached walk-forward predictions (wf_predict.py) for the main tier (day move > 0.6 ATR)
and the session tier (quiet day, 6h move > 3 H1-ATR or 24h move > 7 H1-ATR).
usage: python reference_v2.py <out dir>
"""
import sys, json
import numpy as np, pandas as pd
from activity_recovery import run, report, data_weeks, S

CFG = dict(core_k=1.0, veto=0.03, margin=0.04, min_k=0.6, margin2=0.04, idle_days=1.5, idle_span=2.0, relax=0.04,
           rec="prop", K=15.0, cap=1.5, unit=6.0)
out = sys.argv[1]
A = pd.read_parquet(f"{S}/P_g06_h8.parquet")
A["tier2"] = False
B = pd.read_parquet(f"{S}/P_seconly_h8.parquet")
B["tier2"] = True
P = pd.concat([A, B]).sort_values(["ent", "dir"])
W = data_weeks()
res = {"config": CFG, "variants": []}
for lab, kw in [("v1.0 (main tier only, fixed lot)", dict(margin2=99, idle_days=99, rec="none")),
                ("v1.1 activity (session tier + governor), fixed lot", dict(rec="none")),
                ("v1.1 activity + smooth recovery (default)", {})]:
    T = run(P, **{**CFG, **kw})
    res["variants"].append(report(T, W, lab))
    res["variants"].append(report(T, W, lab + " | last 18m", since="2025-04-24"))
    if "default" in lab:
        T.to_csv(f"{out}/trades_v11.csv")
        res["yearly"] = {int(y): {"trades": len(q), "net": round(q.money.sum(), 1),
                                  "pf": round(q.money[q.money > 0].sum() / max(-q.money[q.money < 0].sum(), 1e-9), 2)}
                         for y, q in T.groupby(T.index.year)}
json.dump(res, open(f"{out}/summary_v11.json", "w"), indent=1, default=str)
pd.set_option("display.width", 250)
print(pd.DataFrame(res["variants"]).to_string(index=False))
print(res["yearly"])
