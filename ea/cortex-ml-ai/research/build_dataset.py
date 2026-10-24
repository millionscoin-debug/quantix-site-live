"""Assemble one UTC XAUUSD M1 series 2020-01 -> 2026-10 from the public sources.

Sources (all public GitHub repos):
  A parshant65-lang/gold-xauusd-1m-2020-2025-historical-data  M1 spot, broker server time (NY+7)
  B getdata-finance/xauusd-1m-ohlcv-metals-historical-data    M1 spot, UTC (weekly rolling sample, all commits)
  C forexrezaei/xauusd-data (git history)                    M5 spot, UTC
  D shikerkattiyar11-debug/xauusd-gold-data                  H1 COMEX futures GC=F, IST  (only to bridge 2025-10..2026-02)
Output parquet columns: time(UTC) open high low close res(native minutes) src
"""
import glob, subprocess, sys, numpy as np, pandas as pd

OUT = sys.argv[1]
A_DIR = "/home/user/parshant65-lang/gold-xauusd-1m-2020-2025-historical-data/XAUUSD/1m"
B_REPO = "/home/user/getdata-finance/xauusd-1m-ohlcv-metals-historical-data"
C_CSV, D_CSV = sys.argv[2], sys.argv[3]

def us_dst(ts):
    """True when US daylight saving is active (second Sun of Mar .. first Sun of Nov)."""
    y = ts.year
    mar = pd.Timestamp(y, 3, 8) + pd.Timedelta(days=(6 - pd.Timestamp(y, 3, 8).weekday()) % 7)
    nov = pd.Timestamp(y, 11, 1) + pd.Timedelta(days=(6 - pd.Timestamp(y, 11, 1).weekday()) % 7)
    return mar <= ts < nov

# ---- A: server time -> UTC (server = UTC+3 in US DST, UTC+2 otherwise)
fs = sorted(glob.glob(f"{A_DIR}/*/*/*.csv"))
a = pd.concat([pd.read_csv(f, usecols=["datetime_utc", "open", "high", "low", "close"]) for f in fs])
a["time"] = pd.to_datetime(a.pop("datetime_utc"))
days = a["time"].dt.normalize()
uniq = pd.Series(days.unique())
off = pd.Series(np.where([us_dst(d) for d in uniq], 3, 2), index=uniq)
a["time"] = a["time"] - pd.to_timedelta(days.map(off).values, unit="h")
a["res"], a["src"] = 1, "A"

# ---- B: union of every weekly sample commit (newer wins)
log = subprocess.run(["git", "-C", B_REPO, "log", "--format=%H"], capture_output=True, text=True).stdout.split()
parts = []
for i, h in enumerate(reversed(log)):
    raw = subprocess.run(["git", "-C", B_REPO, "show", f"{h}:XAUUSD_1m.csv"], capture_output=True, text=True).stdout
    from io import StringIO
    p = pd.read_csv(StringIO(raw)); p["ord"] = i; parts.append(p)
b = pd.concat(parts)
b["time"] = pd.to_datetime(b["datetime"], utc=True).dt.tz_localize(None)
b = b.sort_values(["time", "ord"]).drop_duplicates("time", keep="last")[["time", "open", "high", "low", "close"]]
b["res"], b["src"] = 1, "B"

# ---- C: M5 (UTC)
c = pd.read_csv(C_CSV, parse_dates=["time"]); c["res"], c["src"] = 5, "C"
# ---- D: H1 futures, IST -> UTC
d = pd.read_csv(D_CSV)
d["time"] = pd.to_datetime(d["Datetime"]) - pd.Timedelta(hours=5, minutes=30)
d = d.rename(columns=str.lower)[["time", "open", "high", "low", "close"]]; d["res"], d["src"] = 60, "D"

def rets(df, rule):
    s = df.set_index("time")["close"].resample(rule).last().dropna()
    return s
def best_shift(x, y, rule, shifts):
    """correlate returns of x and y (resampled) for hour shifts applied to y."""
    out = []
    rx = rets(x, rule).diff()
    for sh in shifts:
        yy = y.copy(); yy["time"] = yy["time"] + pd.Timedelta(hours=sh)
        ry = rets(yy, rule).diff()
        j = pd.concat([rx, ry], axis=1, join="inner").dropna()
        out.append((sh, round(j.corr().iloc[0, 1], 3), len(j)))
    return out
print("A vs D (H1) shift corr:", best_shift(a[a.time > "2025-07-15"], d[d.time < "2025-10-01"], "1h", [-2, -1, 0, 1, 2]))
print("B vs D (H1) shift corr:", best_shift(b, d[d.time > "2026-02-02"], "1h", [-1, 0, 1]))
print("B vs C (5m) shift corr:", best_shift(b[b.time > "2026-09-05"], c[c.time < "2026-09-26"], "5min", [-1, 0, 1]))

# basis futures - spot (H1 closes) on the overlaps, used to de-bias D in the gap
def basis(spot, fut):
    s = spot.set_index("time")["close"].resample("1h").last()
    f = fut.set_index("time")["close"]
    j = pd.concat([s, f], axis=1, join="inner").dropna(); j.columns = ["s", "f"]
    return (j.f - j.s)
b1, b2 = basis(a, d), basis(b, d)
print("basis A-overlap mean %.2f sd %.2f | B-overlap first month mean %.2f" % (b1.mean(), b1.std(), b2[:"2026-03-01"].mean()))
cs = pd.concat([basis(c, d)])
print("C vs B overlap price diff: ", (rets(c[c.time < "2026-09-26"], "5min") - rets(b[b.time > "2026-09-05"], "5min")).dropna().describe()[["mean", "std"]].round(3).to_dict())

# D fills every hour (from 2025-07-14 up to the start of B) where A has < 30 native minutes.
# Futures basis is measured per day on the A overlap and on the first weeks of B, interpolated between.
b_start = b.time.min()
bas = pd.concat([b1, b2]).sort_index()
bas = bas[(bas - bas.rolling(48, min_periods=5, center=True).median()).abs() < 6]   # drop roll-jump outliers
bas_d = bas.resample("1D").median()
bas_d = bas_d.reindex(pd.date_range(d.time.min().normalize(), b_start.normalize() + pd.Timedelta(days=1))).interpolate().ffill().bfill()
a_hour_cnt = a.groupby(a.time.dt.floor("1h")).size()
dg = d[(d.time < b_start)].copy()
dg = dg[dg.time.map(a_hour_cnt).fillna(0).values < 30]
adj = dg.time.dt.normalize().map(bas_d).values
for k in ["open", "high", "low", "close"]: dg[k] = dg[k] - adj
a = a[~a.time.dt.floor("1h").isin(set(dg.time))]
cg = c[c.time > b.time.max()]
full = pd.concat([a, dg, b, cg]).sort_values("time").drop_duplicates("time", keep="first")
wd, hr = full.time.dt.weekday, full.time.dt.hour
full = full[(wd < 5) | ((wd == 6) & (hr >= 21))]
full = full[~((full.time.dt.weekday == 4) & (full.time.dt.hour >= 21))]
# monthly coverage report (minutes of native data / expected)
cov = full.groupby([full.time.dt.to_period("M"), "src"]).res.sum().unstack(fill_value=0)
print(cov.to_string())
full = full.reset_index(drop=True)
full.to_parquet(OUT)
print(full.groupby("src").time.agg(["min", "max", "count"]))
