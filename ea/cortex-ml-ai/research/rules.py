"""Evaluate rule-based entry masks on the candidate tables with one-position-at-a-time execution."""
import sys
import numpy as np, pandas as pd
import engine as E

S = "/tmp/claude-0/-home-user-quantix-site-live/d63ac0eb-0794-5f04-8945-165b7e1ba43f/scratchpad/data"


def load(name):
    D = pd.read_parquet(f"{S}/{name}.parquet")
    D = D.sort_values(["ent", "dir"], kind="stable")
    return D


def run(D, mask, label="", max_pos=1, years=range(2020, 2027), show=True):
    take = np.asarray(mask, dtype=bool) & (D.res.values <= 5) & D.valid.values
    done = E.portfolio(take, D.ent.values, D.ex.values, D.pnl.values, max_pos)
    T = D[done]
    out = {}
    for y in years:
        q = T[T.index.year == y]
        out[y] = (round(q.pnl.sum()), len(q), round((q.pnl > 0).mean() * 100) if len(q) else 0)
    last = T[T.index >= "2025-04-24"]
    st = E.stats(last.pnl.values, last.index)
    if show:
        print(f"{label:38s}", " ".join(f"{y}:{v[0]:>6}/{v[1]:<4}" for y, v in out.items()),
              "| L18m net", st.get("net"), "pf", st.get("pf"), "wr", st.get("wr"), "dd", st.get("maxdd"))
    return T, out, st


if __name__ == "__main__":
    for name in sys.argv[1:]:
        D = load(name)
        print("==", name)
        tr = D.d_e2050 > 0
        tr2 = (D.d_e2050 > 0) & (D.d_d50 > 0)
        run(D, np.ones(len(D), bool), "all")
        run(D, tr, "d1 ema20>50")
        run(D, tr2, "d1 ema20>50 & c>ema50")
        run(D, tr & (D.s_rsi < 40), "d1 trend & m15 rsi<40")
        run(D, tr & (D.h_rsi < 45), "d1 trend & h1 rsi<45")
        run(D, tr & (D.h_d20 < 0), "d1 trend & below h1 ema20")
        run(D, tr & (D.d_today < 0), "d1 trend & red day so far")
        run(D, tr & (D.f_sl50 > 0), "d1 trend & h4 slope ok")
        run(D, tr & (D.f_sl50 > 0) & (D.s_rsi < 40), "d1+h4 trend & m15 rsi<40")
        run(D, tr & (D.f_sl50 > 0) & (D.h_d20 < 0), "d1+h4 trend & below h1 ema20")
