"""Charts + tables for the README from the reference simulation outputs."""
import json, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

S = "/tmp/claude-0/-home-user-quantix-site-live/d63ac0eb-0794-5f04-8945-165b7e1ba43f/scratchpad"
OUT = sys.argv[1]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, RED, GRAY = "#2a78d6", "#e34948", "#c9c8c2"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF, "font.family": "DejaVu Sans"})

T = pd.read_csv(f"{S}/ref/trades.csv", index_col=0, parse_dates=True)
eq = T.pnl.cumsum()
gap = (pd.Timestamp("2025-07-14"), pd.Timestamp("2026-02-02"))


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)


# 1) full walk-forward equity
fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
ax.axvspan(*gap, color=GRAY, alpha=0.35, lw=0)
ax.text(gap[0] + (gap[1] - gap[0]) / 2, eq.max() * 0.08, "M1 data mostly missing\n(H1 only, not simulated)", ha="center",
        color=INK2, fontsize=8)
ax.axvline(pd.Timestamp("2025-04-24"), color=INK2, lw=0.8, ls=(0, (3, 3)))
ax.text(pd.Timestamp("2025-05-05"), eq.max() * 0.95, "last 18 months", color=INK2, fontsize=8)
ax.plot(eq.index, eq.values, color=BLUE, lw=2)
ax.set_title("Cortex ML AI - walk-forward equity, $ per 0.01 lot (XAUUSD, 2021-01 .. 2026-10)", color=INK, loc="left")
ax.set_ylabel("cumulative P/L, $")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
style(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/equity_walkforward.png")

# 2) last 18 months, trade by trade
L = T[T.index >= "2025-04-24"]
fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
le = L.pnl.cumsum()
ax.plot(np.arange(len(le)), le.values, color=BLUE, lw=2)
k = int((L.index < gap[1]).sum())
ax.axvline(k, color=INK2, lw=0.8, ls=(0, (3, 3)))
ax.text(k + 4, le.max() * 0.05, "2026-02: data resumes", color=INK2, fontsize=8)
ax.set_title("Last 18 months (2025-04-24 .. 2026-10-23), trade by trade, $ per 0.01 lot", color=INK, loc="left")
ax.set_xlabel("trade #")
ax.set_ylabel("cumulative P/L, $")
style(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/equity_last18m.png")

# 3) monthly P/L (polarity: blue / red), 2025-04 .. 2026-10, one slot per calendar month
cnt = T.pnl.groupby(T.index.to_period("M")).size()
sm = T.pnl.groupby(T.index.to_period("M")).sum()
idx = pd.period_range("2025-04", "2026-10", freq="M")
mo = pd.Series({p: (sm[p] if cnt.get(p, 0) > 0 else np.nan) for p in idx})
fig, ax = plt.subplots(figsize=(10, 3.6), dpi=150)
xs = np.arange(len(mo))
vals = mo.fillna(0).values
ax.bar(xs, vals, width=0.7, color=[BLUE if v >= 0 else RED for v in vals])
ax.axhline(0, color=INK2, lw=0.8)
for x, v, raw in zip(xs, vals, mo.values):
    if np.isnan(raw):
        ax.text(x, 3, "n/d", ha="center", va="bottom", fontsize=7, color=INK2)
    else:
        ax.text(x, v + (3 if v >= 0 else -3), f"{v:+.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7, color=INK2)
ax.set_xticks(xs)
ax.set_xticklabels([str(p) for p in idx], rotation=45, ha="right", fontsize=8)
ax.set_title("Monthly P/L, $ per 0.01 lot  (n/d = no M1 data in the public set; 2025-04 starts on the 24th)", color=INK, loc="left")
style(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/monthly_last18m.png")

# tables
rows = []
for tag, lab in [("ref", "Shield 5 / 4 / 3 (default)"), ("ref_t6_4", "Balanced 6 / 4 / 3"), ("ref_t8_5", "Runner 8 / 5 / 3"),
                 ("ref_t99_3", "Trailing off (TP 13.5 / SL 30)"), ("ref_s40", "Shield, spread 0.40 stress")]:
    r = json.load(open(f"{S}/{tag}/summary.json"))
    a, b = r["periods"]
    rows.append({"profile": lab, "OOS net": a["net"], "OOS PF": a["pf"], "OOS WR%": a["wr"], "OOS maxDD": a["maxdd"],
                 "18m net": b["net"], "18m trades": b["n"], "18m PF": b["pf"], "18m WR%": b["wr"], "18m maxDD": b["maxdd"],
                 "18m R2": b["r2"], "avg win": b["avg_win"], "avg loss": b["avg_loss"]})
P = pd.DataFrame(rows)
P.to_csv(f"{OUT}/profiles.csv", index=False)
r = json.load(open(f"{S}/ref/summary.json"))
Y = pd.DataFrame(r["yearly"]).T[["n", "net", "pf", "wr", "maxdd", "avg_win", "avg_loss", "tp_exits", "sl_exits", "trail_exits"]]
Y.to_csv(f"{OUT}/yearly.csv")
print(P.to_markdown(index=False))
print(Y.to_markdown())
json.dump(r["periods"], open(f"{OUT}/periods.json", "w"), indent=1)
