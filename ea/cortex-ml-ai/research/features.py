"""Feature set computed exactly the way the EA computes it (see CortexFeatures.mqh).

Everything is scale free (divided by ATR or bounded oscillators) except the two
barrier-scale features tp_atr / sl_atr, which tell the model how large the fixed
dollar targets are compared with the current volatility.
"""
import numpy as np
import pandas as pd

SIGNAL_TF = "15min"


def to_server(t):
    """UTC -> broker server time (NY+7: UTC+3 during US DST, UTC+2 otherwise), as MT5/IC Markets."""
    t = pd.DatetimeIndex(t)
    y = t.year
    mar = pd.to_datetime([f"{v}-03-08" for v in y])
    mar = mar + pd.to_timedelta((6 - mar.weekday) % 7, unit="D") + pd.Timedelta(hours=7)
    nov = pd.to_datetime([f"{v}-11-01" for v in y])
    nov = nov + pd.to_timedelta((6 - nov.weekday) % 7, unit="D") + pd.Timedelta(hours=6)
    dst = (t >= mar) & (t < nov)
    return t + pd.to_timedelta(np.where(dst, 3, 2), unit="h")


def resample(m1, rule):
    """bars in broker server time (column stime), exactly like the MT5 chart."""
    g = m1.set_index("stime")
    r = g.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "res": "max"})
    return r.dropna(subset=["open"])


def sma(x, n):
    return pd.Series(x).rolling(n, min_periods=n).mean().values


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def atr(h, l, c, n):
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return sma(tr, n)


def rsi(c, n):
    d = np.diff(c, prepend=c[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    # Wilder smoothing == ewm alpha 1/n
    au = pd.Series(up).ewm(alpha=1.0 / n, adjust=False).mean().values
    ad = pd.Series(dn).ewm(alpha=1.0 / n, adjust=False).mean().values
    rs = au / np.maximum(ad, 1e-12)
    return 100 - 100 / (1 + rs)


def eff_ratio(c, n):
    s = pd.Series(c)
    net = (s - s.shift(n)).abs()
    path = s.diff().abs().rolling(n).sum()
    return (net / path.replace(0, np.nan)).values


def hh(x, n):
    return pd.Series(x).rolling(n, min_periods=n).max().values


def ll(x, n):
    return pd.Series(x).rolling(n, min_periods=n).min().values


def tf_block(b, prefix):
    """generic indicator block on a bar frame b (open, high, low, close)."""
    o, h, l, c = (b[k].values for k in ["open", "high", "low", "close"])
    a14 = atr(h, l, c, 14)
    out = {}
    out[f"{prefix}atr"] = a14
    for n in (1, 3, 6, 12, 24):
        out[f"{prefix}ret{n}"] = (c - np.roll(c, n)) / a14
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    out[f"{prefix}d20"] = (c - e20) / a14
    out[f"{prefix}d50"] = (c - e50) / a14
    out[f"{prefix}d200"] = (c - e200) / a14
    out[f"{prefix}sl50"] = (e50 - np.roll(e50, 5)) / a14
    out[f"{prefix}rsi"] = rsi(c, 14)
    out[f"{prefix}rsi_f"] = rsi(c, 5)
    m20, s20 = sma(c, 20), pd.Series(c).rolling(20).std(ddof=0).values
    out[f"{prefix}bbz"] = (c - m20) / np.maximum(s20, 1e-9)
    out[f"{prefix}bbw"] = 4 * s20 / a14
    hi20, lo20 = hh(h, 20), ll(l, 20)
    out[f"{prefix}pos20"] = (c - lo20) / np.maximum(hi20 - lo20, 1e-9)
    out[f"{prefix}er"] = eff_ratio(c, 20)
    out[f"{prefix}volr"] = a14 / sma(a14, 100)
    rng = np.maximum(h - l, 1e-9)
    out[f"{prefix}body"] = (c - o) / rng
    out[f"{prefix}uw"] = (h - np.maximum(o, c)) / rng
    out[f"{prefix}lw"] = (np.minimum(o, c) - l) / rng
    return pd.DataFrame(out, index=b.index)


def build(m1, tp=13.5, sl=30.0):
    if "stime" not in m1:
        m1 = m1.assign(stime=to_server(m1.time.values))
    s = resample(m1, SIGNAL_TF)
    h1 = resample(m1, "1h")
    h4 = resample(m1, "4h")
    d1 = resample(m1, "1D")
    fs = tf_block(s, "s_")
    fh = tf_block(h1, "h_")
    f4 = tf_block(h4, "f_")[["f_ret1", "f_ret3", "f_d20", "f_d50", "f_rsi", "f_er", "f_pos20", "f_sl50"]]
    # higher TF values known at the signal bar close: shift by one completed bar
    def align(f, rule):
        f = f.copy()
        f.index = f.index + pd.Timedelta(rule)            # a bar's values become known at its close
        return f.reindex(fs.index + pd.Timedelta(SIGNAL_TF), method="ffill").set_axis(fs.index)
    fh_a = align(fh, "1h")
    f4_a = align(f4, "4h")
    fd = tf_block(d1, "d_")[["d_atr", "d_ret1", "d_ret3", "d_ret12", "d_d20", "d_d50", "d_sl50", "d_rsi", "d_er", "d_pos20"]]
    fd["d_e2050"] = (ema(d1.close.values, 20) - ema(d1.close.values, 50)) / fd["d_atr"].values
    fd_a = align(fd, "1D")
    fd_a["d_today"] = (s["close"].values - s["open"].groupby(s.index.normalize()).transform("first").values) / fd_a["d_atr"].values
    fd_a = fd_a.drop(columns=["d_atr"])
    f4_a = pd.concat([f4_a, fd_a], axis=1)
    # daily context: previous day high/low/close and today's open
    dprev = d1[["high", "low", "close"]].shift(1)
    dprev.index = dprev.index
    day = fs.index.normalize()
    pdh = dprev["high"].reindex(day).values
    pdl = dprev["low"].reindex(day).values
    c = s["close"].values
    sa = fs["s_atr"].values
    ha = fh_a["h_atr"].values
    X = pd.concat([fs, fh_a, f4_a], axis=1)
    X["pdh"] = (c - pdh) / ha
    X["pdl"] = (c - pdl) / ha
    # intraday high/low so far (server day)
    sess = s.index.normalize()
    g = pd.DataFrame({"h": s["high"].values, "l": s["low"].values, "k": sess}, index=s.index)
    X["dhi"] = (g.groupby("k").h.cummax().values - c) / ha
    X["dlo"] = (c - g.groupby("k").l.cummin().values) / ha
    X["tp_atr"] = tp / ha
    X["sl_atr"] = sl / ha
    X["s_atr_h"] = sa / ha
    X["hour"] = s.index.hour + s.index.minute / 60.0
    X["dow"] = s.index.weekday
    X["res"] = s["res"].values
    X["close"] = c
    X = X.drop(columns=["s_atr", "h_atr"])
    return X


# directional features flip sign for a short candidate so one model can score both sides
def directional_cols(cols):
    out = []
    for k in cols:
        base = k.split("_", 1)[-1]
        if any(base.startswith(p) for p in ("ret", "d20", "d50", "d200", "sl50", "bbz", "body", "e2050", "today")):
            out.append(k)
        elif k in ("pdh", "pdl", "dhi", "dlo"):
            out.append(k)
    return out


def mirror(X, direction):
    """return feature frame as seen from a candidate of the given direction (+1/-1)."""
    Y = X.copy()
    if direction > 0:
        return Y
    for k in Y.columns:
        base = k.split("_", 1)[-1]
        if any(base.startswith(p) for p in ("ret", "d20", "d50", "d200", "sl50", "bbz", "body", "e2050", "today")):
            Y[k] = -Y[k]
        elif base in ("rsi", "rsi_f"):
            Y[k] = 100 - Y[k]
        elif base == "pos20":
            Y[k] = 1 - Y[k]
    pre = [p for p in ("s_", "h_", "f_") if f"{p}uw" in Y.columns]
    for p in pre:
        Y[f"{p}uw"], Y[f"{p}lw"] = X[f"{p}lw"], X[f"{p}uw"]
    Y["pdh"], Y["pdl"] = -X["pdl"], -X["pdh"]
    Y["dhi"], Y["dlo"] = X["dlo"], X["dhi"]
    return Y
