"""Does recent shadow performance of a setup predict its next performance?
If yes, an online expert-selection layer (bandit / hedge) can follow regimes.

Experts = setup condition x side.  Score = decayed mean P/L of the expert's shadow
trades that CLOSED before the decision time.  Each day we trade the top experts."""
import sys
import numpy as np, pandas as pd
import engine as E

D = pd.read_parquet(sys.argv[1])
D = D[D.res <= 5].copy()
m1 = E.load_m1()
tt = m1.stime.values
D["exit_time"] = tt[D.ex.values]
H = D.hour
conds = {
    "all": np.ones(len(D), bool),
    "trend4": D.f_sl50 > 0.3,
    "trend1": D.h_sl50 > 0.3,
    "ctr4": D.f_sl50 < -0.3,
    "pull_rsi35": D.s_rsi < 35,
    "push_rsi65": D.s_rsi > 65,
    "fade_m15": D.s_ret1 < -1.2,
    "mom_m15": D.s_ret1 > 1.2,
    "bb_low": D.s_bbz < -2,
    "bb_high": D.s_bbz > 2,
    "h1_low": D.h_pos20 < 0.15,
    "h1_high": D.h_pos20 > 0.85,
    "range_fade": (D.f_er < 0.25) & (D.s_bbz < -1.5),
    "trend_pull": (D.f_sl50 > 0.3) & (D.s_rsi < 40),
    "asia": (H >= 23) | (H < 6),
    "london": (H >= 7) & (H < 12),
    "ny": (H >= 12.5) & (H < 17),
    "brk20": D.s_pos20 > 0.97,
    "pdh_near": D.pdh.abs() < 0.3,
    "pdl_near": D.pdl.abs() < 0.3,
}
experts = []
for k, m in conds.items():
    for side in (1, -1):
        experts.append((f"{k}{'L' if side > 0 else 'S'}", np.asarray(m) & (D.dir.values == side)))
days = pd.date_range("2021-01-01", "2026-10-23", freq="B")
P = D.pnl.values
ET = D.exit_time.values
IT = D.index.values
res = {w: [] for w in (5, 10, 20, 40)}
base = []
for name, m in experts:
    pass
# precompute per expert arrays
ex_data = []
for name, m in experts:
    ex_data.append((name, IT[m], ET[m], P[m]))
for d0 in days:
    d1 = d0 + pd.Timedelta(days=1)
    t0, t1 = np.datetime64(d0), np.datetime64(d1)
    today = []
    for name, it, et, p in ex_data:
        sel = (it >= t0) & (it < t1)
        if sel.sum() == 0:
            continue
        nxt = p[sel].mean()
        sc = {}
        for w in res:
            past = (et < t0) & (it >= t0 - np.timedelta64(w, "D"))
            sc[w] = p[past].mean() if past.sum() >= 20 else np.nan
        today.append((name, nxt, sc))
    if not today:
        continue
    base.append(np.mean([x[1] for x in today]))
    for w in res:
        sc = [(x[2][w], x[1]) for x in today if not np.isnan(x[2][w])]
        if len(sc) < 5:
            continue
        sc.sort(key=lambda z: -z[0])
        top = [z[1] for z in sc[:3] if z[0] > 0]
        bot = [z[1] for z in sc[-3:]]
        res[w].append((d0, np.mean(top) if top else np.nan, np.mean(bot)))
print("base mean next-day expert P/L %.3f" % np.mean(base))
for w, r in res.items():
    R = pd.DataFrame(r, columns=["d", "top", "bot"]).set_index("d")
    yr = R.groupby(R.index.year).mean().round(2)
    print(f"window {w}d  top3 %.3f  bottom3 %.3f" % (R.top.mean(), R.bot.mean()))
    print(yr.T.to_string())
