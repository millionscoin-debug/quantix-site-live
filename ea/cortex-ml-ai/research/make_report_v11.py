"""Charts for v1.1 (two models + activity governor + smooth recovery) vs v1.0."""
import sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from activity_recovery import run, data_weeks, S
from reference_v2 import CFG

OUT = sys.argv[1]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, RED, GRAY = "#2a78d6", "#eb6834", "#e34948", "#c9c8c2"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF})
A = pd.read_parquet(f"{S}/P_g06_h8.parquet"); A["tier2"] = False
B = pd.read_parquet(f"{S}/P_seconly_h8.parquet"); B["tier2"] = True
P = pd.concat([A, B]).sort_values(["ent", "dir"])
v10 = run(P, **{**CFG, "margin2": 99, "idle_days": 99, "rec": "none"})
v11 = run(P, **CFG)
gap = (pd.Timestamp("2025-07-14"), pd.Timestamp("2026-02-02"))


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)


fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
ax.axvspan(*gap, color=GRAY, alpha=0.35, lw=0)
e10, e11 = v10.money.cumsum(), v11.money.cumsum()
ax.text(gap[0] + (gap[1] - gap[0]) / 2, e11.max() * 0.06, "M1 data mostly missing\n(not simulated)", ha="center", color=INK2, fontsize=8)
ax.plot(e10.index, e10.values, color=ORANGE, lw=2, label="v1.0: day tier only, fixed lot")
ax.plot(e11.index, e11.values, color=BLUE, lw=2, label="v1.1: + session tier, activity governor, smooth recovery")
ax.text(e11.index[-1], e11.values[-1], f"  v1.1 {e11.values[-1]:+.0f}", color=INK, va="center", fontsize=8)
ax.text(e10.index[-1], e10.values[-1], f"  v1.0 {e10.values[-1]:+.0f}", color=INK, va="center", fontsize=8)
ax.legend(frameon=False, loc="upper left", fontsize=8)
ax.set_title("Cortex ML AI walk-forward equity, $ at base lot 0.01 (XAUUSD 2021-01 .. 2026-10)", color=INK, loc="left")
ax.set_ylabel("cumulative P/L, $")
ax.set_xlim(right=pd.Timestamp("2027-03-01"))
ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
style(ax); fig.tight_layout(); fig.savefig(f"{OUT}/equity_walkforward.png"); plt.close(fig)

L = v11[v11.index >= "2025-04-24"]
fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
le = L.money.cumsum()
ax.plot(np.arange(len(le)), le.values, color=BLUE, lw=2)
k = int((L.index < gap[1]).sum())
ax.axvline(k, color=INK2, lw=0.8, ls=(0, (3, 3)))
ax.text(k + 4, le.max() * 0.05, "2026-02: data resumes", color=INK2, fontsize=8)
ax.set_title("v1.1, last 18 months (2025-04-24 .. 2026-10-23), trade by trade, $ at base lot 0.01", color=INK, loc="left")
ax.set_xlabel("trade #"); ax.set_ylabel("cumulative P/L, $")
style(ax); fig.tight_layout(); fig.savefig(f"{OUT}/equity_last18m.png"); plt.close(fig)

idx = pd.period_range("2025-04", "2026-10", freq="M")
cnt = L.money.groupby(L.index.to_period("M")).size(); sm = L.money.groupby(L.index.to_period("M")).sum()
mo = pd.Series({p: (sm[p] if cnt.get(p, 0) > 0 else np.nan) for p in idx})
fig, ax = plt.subplots(figsize=(10, 3.6), dpi=150)
xs = np.arange(len(mo)); vals = mo.fillna(0).values
ax.bar(xs, vals, width=0.7, color=[BLUE if v >= 0 else RED for v in vals]); ax.axhline(0, color=INK2, lw=0.8)
for x, v, raw in zip(xs, vals, mo.values):
    lab = "n/d" if np.isnan(raw) else f"{v:+.0f}"
    ax.text(x, (3 if v >= 0 else v - 3) if not np.isnan(raw) else 3, lab, ha="center", va="bottom" if v >= 0 else "top", fontsize=7, color=INK2)
ax.set_xticks(xs); ax.set_xticklabels([str(p) for p in idx], rotation=45, ha="right", fontsize=8)
ax.set_title("v1.1 monthly P/L, $ at base lot 0.01  (n/d = no M1 data in the public set)", color=INK, loc="left")
style(ax); fig.tight_layout(); fig.savefig(f"{OUT}/monthly_last18m.png"); plt.close(fig)

W = sorted(w for w in data_weeks() if w.start_time >= pd.Timestamp("2025-04-21"))
wk = L.groupby(L.index.to_period("W")).size()
c11 = [wk.get(w, 0) for w in W]
wk0 = v10[v10.index >= "2025-04-24"].groupby(v10[v10.index >= "2025-04-24"].index.to_period("W")).size()
c10 = [wk0.get(w, 0) for w in W]
fig, ax = plt.subplots(figsize=(10, 3.4), dpi=150)
x = np.arange(len(W))
ax.bar(x - 0.2, c10, width=0.38, color=ORANGE, label="v1.0")
ax.bar(x + 0.2, c11, width=0.38, color=BLUE, label="v1.1")
ax.set_xticks(x[::3]); ax.set_xticklabels([str(w.start_time.date()) for w in W][::3], rotation=45, ha="right", fontsize=7)
ax.legend(frameon=False, fontsize=8)
ax.set_title("Trades per week, weeks with data in the last 18 months (v1.0 vs v1.1)", color=INK, loc="left")
style(ax); fig.tight_layout(); fig.savefig(f"{OUT}/trades_per_week.png"); plt.close(fig)
print("zero weeks v1.0:", sum(c == 0 for c in c10), "v1.1:", sum(c == 0 for c in c11), "of", len(W))
