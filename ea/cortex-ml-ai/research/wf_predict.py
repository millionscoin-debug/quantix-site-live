"""Monthly walk-forward logistic predictions for every gated candidate (the EA's model), cached.

usage: python wf_predict.py <cand table> <min gate> <hour_from> <hour_to> <out.parquet>
"""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
import engine as E
from rules import load


def predict(name, gate, h0, h1, train_days=1095, C=0.05, secondary=False, only_secondary=False):
    D = load(name)
    hr = D.index.hour + D.index.minute / 60
    g = D.d_today > gate
    if only_secondary:
        g = (D.d_today.abs() <= gate) & ((D.h_ret6 > 3.0) | (D.h_ret24 > 7.0))
    elif secondary:
        # session momentum on days where the main engine is silent (|day move| <= gate)
        g = g | ((D.d_today.abs() <= gate) & ((D.h_ret6 > 3.0) | (D.h_ret24 > 7.0)))
    D = D[g & (hr >= h0) & (hr < h1) & (D.res <= 5) & D.valid].copy()
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
        tr = (et < a64) & (it >= a64 - np.timedelta64(train_days, "D"))
        te = (it >= a64) & (it < b64)
        if te.sum() == 0 or tr.sum() < 300:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        lr = LogisticRegression(C=C, max_iter=500).fit(np.clip((X[tr] - mu) / sd, -4, 4), D.y.values[tr])
        P[te] = lr.predict_proba(np.clip((X[te] - mu) / sd, -4, 4))[:, 1]
        B[te] = D.y.values[tr].mean()
    D["p"], D["base"] = P, B
    D = D.dropna(subset=["p"]).sort_values(["ent", "dir"])
    D["tier2"] = D.d_today.abs() <= gate
    return D[["p", "base", "dir", "ent", "ex", "pnl", "reason", "hold", "d_today", "exit_time", "tier2", "h_ret6", "h_ret24"]]


if __name__ == "__main__":
    name, gate, h0, h1, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    mode = sys.argv[6] if len(sys.argv) > 6 else ""
    P = predict(name, gate, h0, h1, secondary=mode == "sec", only_secondary=mode == "seconly")
    P.to_parquet(out)
    print(len(P), "predictions saved")
