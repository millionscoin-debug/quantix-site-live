"""Stitch the rolling 500-candle XAU/USD M5 window of forexrezaei/xauusd-data
from its git history into one continuous series (2026-09 -> 2026-10)."""
import json, subprocess, sys, pandas as pd

repo, out = sys.argv[1], sys.argv[2]
log = subprocess.run(["git", "-C", repo, "log", "--format=%H %ct"], capture_output=True, text=True).stdout.split("\n")
commits = [(l.split()[0], int(l.split()[1])) for l in log if l.strip()]
picked, last = [], None
for h, t in commits:                      # newest -> oldest, keep one per ~20h
    if last is None or last - t >= 20 * 3600:
        picked.append(h); last = t
rows = {}
for h in reversed(picked):                # oldest first, newer overwrite
    try:
        raw = subprocess.run(["git", "-C", repo, "show", f"{h}:xauusd-m5.json"], capture_output=True, text=True, timeout=120).stdout
        d = json.loads(raw)
    except Exception as e:
        print("skip", h, e); continue
    for c in d.get("candles", []):
        rows[c["time"]] = (c["open"], c["high"], c["low"], c["close"])
df = pd.DataFrame([(k, *v) for k, v in rows.items()], columns=["time", "open", "high", "low", "close"])
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)
df.to_csv(out, index=False)
print(len(picked), "commits,", len(df), "rows", df.time.min(), "->", df.time.max())
