"""Execution options on top of a walk-forward prediction file: max positions, daily SL cap, cooldown."""
import sys
import numpy as np, pandas as pd
import engine as E
from numba import njit

S = "/tmp/claude-0/-home-user-quantix-site-live/d63ac0eb-0794-5f04-8945-165b7e1ba43f/scratchpad/data"


@njit(cache=True)
def execute(take, ent, ex, pnl, reason, day, max_pos, max_sl_day, cool_bars, min_gap_bars):
    m = len(ent)
    done = np.zeros(m, dtype=np.bool_)
    open_ex = np.full(max_pos, -1, dtype=np.int64)
    sl_today = 0
    cur_day = -1
    block_until = -1
    last_entry = -10 ** 12
    # resolved SL events are applied when their exit time has passed
    pend_ex = np.full(m, -1, dtype=np.int64)
    pend_sl = np.zeros(m, dtype=np.bool_)
    npend = 0
    for k in range(m):
        if day[k] != cur_day:
            cur_day = day[k]
            sl_today = 0
        # apply exits that happened before this entry
        j = 0
        while j < npend:
            if pend_ex[j] < ent[k]:
                if pend_sl[j]:
                    sl_today += 1
                    if ent[k] - pend_ex[j] < cool_bars:
                        if pend_ex[j] + cool_bars > block_until:
                            block_until = pend_ex[j] + cool_bars
                pend_ex[j] = pend_ex[npend - 1]
                pend_sl[j] = pend_sl[npend - 1]
                npend -= 1
            else:
                j += 1
        if not take[k]:
            continue
        if sl_today >= max_sl_day or ent[k] < block_until or ent[k] - last_entry < min_gap_bars:
            continue
        slot = -1
        for s in range(max_pos):
            if open_ex[s] < ent[k]:
                slot = s
                break
        if slot < 0:
            continue
        open_ex[slot] = ex[k]
        done[k] = True
        last_entry = ent[k]
        pend_ex[npend] = ex[k]
        pend_sl[npend] = reason[k] == 2
        npend += 1
    return done


def evaluate(P, take, max_pos=1, max_sl_day=99, cool=0, gap=0, label=""):
    day = (P.index.normalize().values.astype("datetime64[D]").astype(np.int64))
    done = execute(np.asarray(take), P.ent.values, P.ex.values, P.pnl.values, P.reason.values.astype(np.int64),
                   day, max_pos, max_sl_day, cool, gap)
    T = P[done]
    by = {y: round(q.pnl.sum()) for y, q in T.groupby(T.index.year)}
    last = T[T.index >= "2025-04-24"]
    st = E.stats(last.pnl.values, last.index)
    print(f"{label:34s} {by} | L18 net {st['net']} n {st['n']} pf {st['pf']} wr {st['wr']} dd {st['maxdd']} r2 {st['r2']} mo+ {st['mo_pos']} worst {st['worst_mo']}")
    return T


if __name__ == "__main__":
    P = pd.read_parquet(f"{S}/{sys.argv[1]}.parquet").sort_values(["ent", "dir"])
    mg = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    take = (P.p > P.base + mg).values
    for mp in (1, 2, 3):
        evaluate(P, take, max_pos=mp, label=f"maxpos {mp}")
        evaluate(P, take, max_pos=mp, max_sl_day=1, label=f"maxpos {mp} 1SL/day")
        evaluate(P, take, max_pos=mp, gap=15, label=f"maxpos {mp} gap15m")
        evaluate(P, take, max_pos=mp, gap=60, label=f"maxpos {mp} gap60m")
        evaluate(P, take, max_pos=mp, cool=240, label=f"maxpos {mp} cool4h after SL")
