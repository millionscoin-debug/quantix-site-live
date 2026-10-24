"""Walk-forward meta-labelling study with LightGBM (upper bound of what an ML filter can do).

Train on every candidate whose trade had closed before the month starts (purged),
predict P(win) for the month, execute with at most one open position.
"""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
import engine as E

D = pd.read_parquet(sys.argv[1])
use_dir = int(sys.argv[2]) if len(sys.argv) > 2 else 1
thr_list = [float(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["0.86", "0.88", "0.9"])]
train_years = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
META = ["ent", "ex", "pnl", "reason", "mfe", "mae", "hold", "close", "res"]
feats = [k for k in D.columns if k not in META and (use_dir or k != "dir")]
m1 = E.load_m1()
tt = m1.stime.values
D = D[D.res <= 5]
D["exit_time"] = tt[D.ex.values]
D["y"] = (D.pnl > 0).astype(int)
months = pd.period_range("2023-01", "2026-10", freq="M")
preds = []
for mo in months:
    a, b = mo.start_time, mo.end_time
    tr = D[(D.exit_time < a) & (D.index >= a - pd.Timedelta(days=365 * train_years))]
    te = D[(D.index >= a) & (D.index <= b)]
    if len(te) == 0:
        continue
    w = np.where(tr.y == 0, 1.0, 1.0)
    mdl = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200,
                             subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, verbose=-1)
    mdl.fit(tr[feats], tr.y, sample_weight=w)
    p = mdl.predict_proba(te[feats])[:, 1]
    preds.append(pd.DataFrame({"p": p}, index=te.index).assign(dir=te.dir.values, ent=te.ent.values,
                                                                ex=te.ex.values, pnl=te.pnl.values))
P = pd.concat(preds).sort_values(["ent", "dir"])
imp = pd.Series(mdl.feature_importances_, index=feats).sort_values(ascending=False)
print("top features:", ", ".join(imp.index[:20]))
from sklearn.metrics import roc_auc_score
for y in sorted(set(P.index.year)):
    q = P[P.index.year == y]
    print(y, "AUC %.3f" % roc_auc_score(q.pnl > 0, q.p), "base wr %.3f" % (q.pnl > 0).mean())
for thr in thr_list:
    take = (P.p.values >= thr)
    done = E.portfolio(take, P.ent.values, P.ex.values, P.pnl.values, 1)
    T = P[done]
    for per, q in [("2023-24", T[T.index < "2025-01-01"]), ("2025-26", T[T.index >= "2025-01-01"]),
                   ("last18m", T[T.index >= "2025-04-24"])]:
        print(f"thr {thr} {per}", E.stats(q.pnl.values, q.index))
P.to_parquet(sys.argv[1].replace(".parquet", f"_wf{use_dir}.parquet"))
