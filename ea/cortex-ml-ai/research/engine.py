"""Core simulation engine shared by every research script.

Prices are in USD per ounce, so 1.0 of price distance == $1 of P/L on 0.01 lot XAUUSD.
All trade-management numbers (SL, TP, trailing) are therefore "dollars per 0.01 lot",
which is exactly how the EA inputs are expressed.

Intrabar path model (same idea as MT5 "1 minute OHLC" generation):
    bullish bar  O -> L -> H -> C        bearish bar  O -> H -> L -> C
Each leg is monotonic, so stop / target / trailing interactions are resolved exactly
for that path.  The series is the BID; ask = bid + spread.
"""
import numpy as np
import pandas as pd
from numba import njit

DATA = "/tmp/claude-0/-home-user-quantix-site-live/d63ac0eb-0794-5f04-8945-165b7e1ba43f/scratchpad/data/xau_full.parquet"


def load_m1():
    """unified M1 frame; 'stime' = broker server time (all bars/candidates are indexed by it)."""
    import features as F
    f = pd.read_parquet(DATA)
    f = f.sort_values("time").reset_index(drop=True)
    f["stime"] = F.to_server(f.time.values)
    return f


# ----------------------------------------------------------------------------------------
# trade management parameters (packed in a float array for numba)
#  0 sl            hard stop distance                       (30)
#  1 tp            take profit distance                     (13.5)
#  2 act           trailing activation profit               (e.g. 7)
#  3 lock          profit locked when trailing activates    (>= 3)
#  4 dist          trailing distance from the peak          (e.g. 4)
#  5 step          minimal SL improvement to modify         (0.5)
#  6 maxhold       max holding time in minutes (0 = none)
#  7 spread        bid/ask spread
#  8 comm          round-turn commission per 0.01 lot
#  9 be_after      minutes after which an unprofitable trade gets a "time decay" SL (0=off)
# 10 tight_k       trailing distance multiplier once price is >= 80% of TP (1 = off)
# ----------------------------------------------------------------------------------------
P_SL, P_TP, P_ACT, P_LOCK, P_DIST, P_STEP, P_MAXH, P_SPR, P_COMM, P_DECAY, P_TIGHT = range(11)


def mgmt(sl=30.0, tp=13.5, act=7.0, lock=3.0, dist=4.0, step=0.5, maxhold=0, spread=0.25,
         comm=0.07, decay=0, tight=1.0):
    return np.array([sl, tp, act, lock, dist, step, maxhold, spread, comm, decay, tight], dtype=np.float64)


@njit(cache=True)
def _leg(direction, p0, p1, entry, slv, tpv, peak, prm, dist):
    """Walk one monotonic leg p0->p1 (prices already in the position's exit-side quote).
    returns (closed, exit_price, slv, peak)."""
    if direction > 0:
        if p1 >= p0:                      # rising leg: target, then trailing ratchet
            if p1 >= tpv:
                return True, tpv, slv, peak
            if p1 > peak:
                peak = p1
                prof = peak - entry
                if prof >= prm[P_ACT]:
                    d = dist
                    if prof >= 0.8 * prm[P_TP]:
                        d = dist * prm[P_TIGHT]
                    ns = max(peak - d, entry + prm[P_LOCK])
                    if ns >= slv + prm[P_STEP]:
                        slv = ns
            return False, 0.0, slv, peak
        else:                             # falling leg: stop
            if p1 <= slv:
                return True, min(slv, p0), slv, peak
            return False, 0.0, slv, peak
    else:
        if p1 <= p0:
            if p1 <= tpv:
                return True, tpv, slv, peak
            if p1 < peak:
                peak = p1
                prof = entry - peak
                if prof >= prm[P_ACT]:
                    d = dist
                    if prof >= 0.8 * prm[P_TP]:
                        d = dist * prm[P_TIGHT]
                    ns = min(peak + d, entry - prm[P_LOCK])
                    if ns <= slv - prm[P_STEP]:
                        slv = ns
            return False, 0.0, slv, peak
        else:
            if p1 >= slv:
                return True, max(slv, p0), slv, peak
            return False, 0.0, slv, peak


@njit(cache=True)
def sim_trade(direction, i0, t, o, h, l, c, prm, tp_dist, trail_dist):
    """Open at the OPEN of bar i0 and manage until exit.
    returns (pnl, exit_index, exit_reason, mfe, mae)
    exit_reason: 1 TP, 2 SL(hard), 3 trailing/locked SL, 4 time, 5 data end"""
    n = len(o)
    spr = prm[P_SPR]
    # quotes on the exit side: long exits on bid, short exits on ask (= bid + spread)
    q = 0.0 if direction > 0 else spr
    entry = o[i0] + (spr if direction > 0 else 0.0)
    slv = entry - direction * prm[P_SL]
    tpv = entry + direction * tp_dist
    peak = entry
    mfe = 0.0
    mae = 0.0
    hard = slv
    t0 = t[i0]
    for i in range(i0, n):
        if prm[P_MAXH] > 0 and (t[i] - t0) > prm[P_MAXH] * 60:
            px = o[i] + q
            return direction * (px - entry) - prm[P_COMM], i, 4, mfe, mae
        oo = o[i] + q
        hh = h[i] + q
        ll = l[i] + q
        cc = c[i] + q
        if i > i0:                         # gap at the open of the bar
            if direction > 0 and oo <= slv:
                r = 2 if slv == hard else 3
                return oo - entry - prm[P_COMM], i, r, mfe, mae
            if direction < 0 and oo >= slv:
                r = 2 if slv == hard else 3
                return entry - oo - prm[P_COMM], i, r, mfe, mae
            if direction > 0 and oo >= tpv:
                return oo - entry - prm[P_COMM], i, 1, mfe, mae
            if direction < 0 and oo <= tpv:
                return entry - oo - prm[P_COMM], i, 1, mfe, mae
        if cc >= oo:
            a1, a2 = ll, hh
        else:
            a1, a2 = hh, ll
        pts = (oo, a1, a2, cc)
        for k in range(3):
            closed, px, slv, peak = _leg(direction, pts[k], pts[k + 1], entry, slv, tpv, peak, prm, trail_dist)
            if closed:
                pnl = direction * (px - entry) - prm[P_COMM]
                if px == tpv:
                    r = 1
                elif slv == hard:
                    r = 2
                else:
                    r = 3
                return pnl, i, r, mfe, mae
        fav = direction * ((hh if direction > 0 else ll) - entry)
        adv = direction * ((ll if direction > 0 else hh) - entry)
        if fav > mfe:
            mfe = fav
        if adv < mae:
            mae = adv
    return direction * (c[n - 1] + q - entry) - prm[P_COMM], n - 1, 5, mfe, mae


@njit(cache=True)
def label_all(idx, dirs, t, o, h, l, c, prm, tp_arr, dist_arr):
    """simulate one trade per candidate (entry at open of m1 index idx[k])."""
    m = len(idx)
    pnl = np.zeros(m)
    ex = np.zeros(m, dtype=np.int64)
    rs = np.zeros(m, dtype=np.int64)
    mfe = np.zeros(m)
    mae = np.zeros(m)
    for k in range(m):
        a, b, r, f1, f2 = sim_trade(dirs[k], idx[k], t, o, h, l, c, prm, tp_arr[k], dist_arr[k])
        pnl[k] = a
        ex[k] = b
        rs[k] = r
        mfe[k] = f1
        mae[k] = f2
    return pnl, ex, rs, mfe, mae


@njit(cache=True)
def portfolio(take, idx, ex, pnl, max_pos):
    """walk candidates in time order; take[k] means the model wants the trade.
    at most max_pos simultaneous positions. returns mask of executed candidates."""
    m = len(idx)
    done = np.zeros(m, dtype=np.bool_)
    open_ex = np.full(max_pos, -1, dtype=np.int64)
    for k in range(m):
        if not take[k]:
            continue
        slot = -1
        for s in range(max_pos):
            if open_ex[s] < idx[k]:
                slot = s
                break
        if slot < 0:
            continue
        open_ex[slot] = ex[k]
        done[k] = True
    return done


def stats(pnl, times, lots=1.0, start_balance=1000.0):
    """pnl in $ per 0.01 lot; lots = multiplier of 0.01 lot."""
    p = np.asarray(pnl) * lots
    if len(p) == 0:
        return {"n": 0}
    eq = start_balance + np.cumsum(p)
    peak = np.maximum.accumulate(np.concatenate([[start_balance], eq]))[1:]
    dd = (peak - eq).max()
    wins = p[p > 0].sum()
    loss = -p[p < 0].sum()
    x = np.arange(len(eq))
    r2 = np.corrcoef(x, eq)[0, 1] ** 2 if len(eq) > 2 else 0
    s = pd.Series(p, index=pd.to_datetime(times))
    mo = s.resample("ME").sum()
    return {
        "n": len(p),
        "net": round(p.sum(), 1),
        "pf": round(wins / loss, 2) if loss > 0 else np.inf,
        "wr": round((p > 0).mean() * 100, 1),
        "avg": round(p.mean(), 2),
        "maxdd": round(dd, 1),
        "ret_dd": round(p.sum() / dd, 2) if dd > 0 else np.inf,
        "r2": round(r2, 3),
        "mo_pos": f"{(mo > 0).sum()}/{len(mo)}",
        "worst_mo": round(mo.min(), 1),
    }
