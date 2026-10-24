"""Activity governor + smooth-recovery study on the walk-forward predictions.

Execution policy (what the EA does at every signal):
  e = p - base
  idle ramp r = clip((days since last entry - idle_days) / idle_span, 0, 1)
  margin_eff = max(margin - relax * r, margin_floor);  core_eff = max(core_k - core_relax * r, min_k)
  take = (d >= core_eff and e > -veto) or (min_k < d < core_eff and e > margin_eff)
Recovery (lot multiplier from REALISED equity known at entry):
  deficit = high-water mark - realised equity  ($ at the base lot)
  prop:  mult = min(cap, 1 + deficit / (K * unit))          (spreads the loss over ~K trades)
  fixed: mult = cap while deficit > 0
"""
import numpy as np, pandas as pd

S = "/tmp/claude-0/-home-user-quantix-site-live/d63ac0eb-0794-5f04-8945-165b7e1ba43f/scratchpad/data"


def data_weeks():
    D = pd.read_parquet(f"{S}/cand_t5_4.parquet", columns=["res", "valid"])
    D = D[(D.res <= 5) & D.valid]
    w = D.groupby(D.index.to_period("W")).size()
    return {k for k in w[w >= 300].index if k.start_time >= pd.Timestamp("2021-01-04")}          # weeks with (almost) full M15 coverage (2 dirs x ~480 bars)


def run(P, core_k=1.0, veto=0.03, margin=0.04, min_k=0.6, max_pos=1, gap_min=0,
        idle_days=99.0, idle_span=3.0, relax=0.0, core_relax=0.0, margin_floor=0.0,
        rec="none", cap=1.0, K=10.0, unit=6.0, margin2=99.0, rec_relax=0.0, rec_pos=0):
    t = P.index.values
    et = P.exit_time.values
    e = (P.p - P.base).values
    d = P.d_today.values
    pnl = P.pnl.values
    t2 = P.tier2.values if "tier2" in P else np.zeros(len(P), bool)
    ent, ex = P.ent.values, P.ex.values
    open_ex = []                       # (ex index, exit time, money)
    pending = []                       # realised later
    eq = hwm = 0.0
    last_entry = None
    rows = []
    for k in range(len(P)):
        # realise trades closed before this signal
        still = []
        for (xe, xt, money) in pending:
            if xt <= t[k]:
                eq += money
                hwm = max(hwm, eq)
            else:
                still.append((xe, xt, money))
        pending = still
        idle = 0.0 if last_entry is None else (t[k] - last_entry) / np.timedelta64(1, "D")
        r = min(1.0, max(0.0, (idle - idle_days) / idle_span)) if idle_days < 90 else 0.0
        under = (hwm - eq) > 0.5
        m_eff = max(margin - relax * r - (rec_relax if under else 0.0), margin_floor)
        c_eff = max(core_k - core_relax * r, min_k)
        if t2[k]:
            take = e[k] > max(margin2 - relax * r - (rec_relax if under else 0.0), margin_floor)
        elif d[k] <= min_k:
            continue
        else:
            take = (d[k] >= c_eff and e[k] > -veto) or (d[k] < c_eff and e[k] > m_eff)
        if not take:
            continue
        busy = sum(1 for x in open_ex if x >= ent[k])
        open_ex = [x for x in open_ex if x >= ent[k]]
        if busy >= (max(max_pos, rec_pos) if under else max_pos):
            continue
        if gap_min and last_entry is not None and (t[k] - last_entry) < np.timedelta64(gap_min, "m"):
            continue
        deficit = hwm - eq
        mult = 1.0
        if rec == "prop" and deficit > 0:
            mult = min(cap, 1.0 + deficit / (K * unit))
        elif rec == "fixed" and deficit > 0:
            mult = cap
        mult = round(mult * 100) / 100.0           # lot step 0.01 on a 1.00 base
        money = pnl[k] * mult
        open_ex.append(ex[k])
        pending.append((ex[k], et[k], money))
        last_entry = t[k]
        rows.append((t[k], et[k], pnl[k], mult, money))
    return pd.DataFrame(rows, columns=["t", "et", "pnl", "mult", "money"]).set_index("t")


def report(T, weeks, label, since=None):
    if since is not None:
        T = T[T.index >= since]
        weeks = {w for w in weeks if w.start_time >= pd.Timestamp(since)}
    m = T.money.values
    eq = np.cumsum(m)
    dd = (np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq).max()
    pf = m[m > 0].sum() / max(-m[m < 0].sum(), 1e-9)
    r2 = np.corrcoef(np.arange(len(eq)), eq)[0, 1] ** 2
    wk = T.groupby(T.index.to_period("W")).size()
    per_week = [wk.get(w, 0) for w in sorted(weeks)]
    zero = np.mean([c == 0 for c in per_week])
    # longest run of consecutive data-weeks with no trade
    run_, best = 0, 0
    for c in per_week:
        run_ = run_ + 1 if c == 0 else 0
        best = max(best, run_)
    # recovery speed: trades needed to make a new high after each losing trade
    hw, need, cur = 0.0, [], None
    for i, v in enumerate(eq):
        if v < hw and cur is None:
            cur = i
        if v >= hw:
            if cur is not None:
                need.append(i - cur)
                cur = None
            hw = v
    return {"policy": label, "trades": len(T), "per_wk": round(np.mean(per_week), 2), "zero_wk%": round(100 * zero, 1),
            "max_zero_wks": best, "net": round(eq[-1], 0) if len(eq) else 0, "pf": round(pf, 2), "maxdd": round(dd, 0),
            "ret/dd": round(eq[-1] / dd, 2) if dd > 0 else np.inf, "r2": round(r2, 3),
            "rec_trades": round(np.mean(need), 1) if need else 0, "avg_mult": round(T.mult.mean(), 2)}


if __name__ == "__main__":
    import sys
    W = data_weeks()
    rows = []
    preds = {k: pd.read_parquet(f"{S}/P_{k}.parquet") for k in ("g06_h8", "g06_h1", "g04_h8", "g04_h1")}
    base = dict()
    pols = [
        ("A current (g0.6 h8-23)", "g06_h8", {}),
        ("B hours 1.5-23", "g06_h1", {}),
        ("C 2 pos, gap 60m", "g06_h8", dict(max_pos=2, gap_min=60)),
        ("D margin 0.02", "g06_h8", dict(margin=0.02)),
        ("E governor: idle>2d relax margin->0", "g06_h8", dict(idle_days=2, idle_span=3, relax=0.04)),
        ("F governor + core 1.0->0.8", "g06_h8", dict(idle_days=2, idle_span=3, relax=0.04, core_relax=0.2)),
        ("G gate0.4 + governor", "g04_h8", dict(min_k=0.4, idle_days=2, idle_span=3, relax=0.04, core_relax=0.2)),
        ("H hours1.5-23 + 2pos gap60 + governor", "g06_h1", dict(max_pos=2, gap_min=60, idle_days=2, idle_span=3, relax=0.04, core_relax=0.2)),
        ("I hours1.5-23 + governor", "g06_h1", dict(idle_days=2, idle_span=3, relax=0.04, core_relax=0.2)),
        ("J 2pos gap60 + governor", "g06_h8", dict(max_pos=2, gap_min=60, idle_days=2, idle_span=3, relax=0.04, core_relax=0.2)),
    ]
    for lab, pk, kw in pols:
        T = run(preds[pk], **kw)
        rows.append(report(T, W, lab))
        rows.append(report(T, W, lab + " | L18", since="2025-04-24"))
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(R.to_string(index=False))
