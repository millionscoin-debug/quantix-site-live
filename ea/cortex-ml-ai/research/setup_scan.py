"""Event-style setups on the candidate table (features are already mirrored to the
candidate's point of view, so 'ret < 0' always means "price moved against the trade direction")."""
import sys
import numpy as np, pandas as pd

D = pd.read_parquet(sys.argv[1])
D = D[D.res <= 5].copy()
D["yr"] = D.index.year
H = D.hour
asia = (H >= 23) | (H < 6)
lon = (H >= 7) & (H < 12)
ny = (H >= 12.5) & (H < 17)
late = (H >= 17) & (H < 21)
up4 = D.f_sl50 > 0.3
dn4 = D.f_sl50 < -0.3
rng4 = D.f_er < 0.25

S = {
    "fade_m15_2atr": D.s_ret1 < -2.0,
    "fade_45m_3atr": D.s_ret3 < -3.0,
    "fade_90m_4atr": D.s_ret6 < -4.0,
    "rsi5<15": D.s_rsi_f < 15,
    "bbz<-2.5": D.s_bbz < -2.5,
    "bbz<-2.5 asia": (D.s_bbz < -2.5) & asia,
    "bbz<-2 range4": (D.s_bbz < -2) & rng4,
    "pullback trend4 rsi<35": up4 & (D.s_rsi < 35),
    "pullback trend4 rsi<30 hrsi>45": up4 & (D.s_rsi < 30) & (D.h_rsi > 45),
    "with trend4 any": up4,
    "against trend4 any": dn4,
    "h1 pullback in trend": up4 & (D.h_d20 < -1.0) & (D.h_d50 > -0.5),
    "breakout pos20=1 er>0.4": (D.s_pos20 > 0.98) & (D.s_er > 0.4),
    "pdl sweep": (D.pdl > -0.3) & (D.pdl < 0.15) & (D.s_lw > 0.4),
    "day low dist<0.2 & lw": (D.dlo < 0.2) & (D.s_lw > 0.5),
    "asia any": asia, "lon any": lon, "ny any": ny, "late any": late,
    "fade_m15 asia": (D.s_ret1 < -1.5) & asia,
    "fade_m15 lon": (D.s_ret1 < -1.5) & lon,
    "fade_m15 ny": (D.s_ret1 < -1.5) & ny,
    "momentum m15 ny": (D.s_ret1 > 2.0) & ny,
    "lowvol tp_atr>1.5 range": (D.tp_atr > 1.5) & rng4,
}
rows = []
for name, m in S.items():
    s = D[m]
    by = s.groupby("yr").pnl.agg(["mean", "size"])
    r = {"setup": name, "n/yr": int(by["size"].mean()), "pos_yrs": int((by["mean"] > 0).sum()), "mean": s.pnl.mean()}
    for y, v in by["mean"].items():
        r[y] = v
    rows.append(r)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).round(2).to_string())
