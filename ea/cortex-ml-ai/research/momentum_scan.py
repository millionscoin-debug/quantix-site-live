"""Intraday-momentum family: trade in the direction of a strong day (move from the server-day
open measured in daily ATR), with timing and context variants."""
import sys
import numpy as np, pandas as pd
from rules import load, run

for name in sys.argv[1:]:
    D = load(name)
    hr = D.index.hour + D.index.minute / 60
    ok = ~((hr >= 23) | (hr < 1.5))                      # no rollover / first 30 min of the day
    print("==", name)
    for k in (0.6, 0.8, 1.0, 1.2):
        run(D, ok & (D.d_today > k), f"day move > {k}")
    for k in (0.8, 1.0):
        for h0, h1 in ((1.5, 10), (10, 16), (16, 23), (8, 20)):
            run(D, ok & (D.d_today > k) & (hr >= h0) & (hr < h1), f"day>{k} hours {h0}-{h1}")
        run(D, ok & (D.d_today > k) & (D.s_rsi < 50), f"day>{k} & m15 pullback rsi<50")
        run(D, ok & (D.d_today > k) & (D.s_d20 < 0), f"day>{k} & below m15 ema20")
        run(D, ok & (D.d_today > k) & (D.d_e2050 > 0), f"day>{k} & d1 trend")
        run(D, ok & (D.d_today > k) & (D.d_e2050 < 0), f"day>{k} & against d1 trend")
        run(D, ok & (D.d_today > k) & (D.dhi < 0.25), f"day>{k} & near day extreme")
        run(D, ok & (D.d_today > k) & (D.dhi > 0.4), f"day>{k} & pulled back from extreme")
