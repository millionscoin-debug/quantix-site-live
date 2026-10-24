"""Line-by-line Python port of the MQL5 feature builder / labeller (CortexML_AI.mq5) used to
verify parity with the research pipeline (features.py + engine.py) on identical data.

The broker gives native M15 and H1 bars; we emulate them by resampling the unified M1 series,
then run the EA algorithms (CTF.Compute, Aggregate, LastClosed, MakeFeatures, SimTrade)."""
import sys
import numpy as np, pandas as pd
from numba import njit
import engine as E
import features as F
from rules import load


class CTF:
    def __init__(self, t, o, h, l, c, dur):
        self.t, self.o, self.h, self.l, self.c, self.dur = t, o, h, l, c, dur
        self.n = len(t)

    def compute(self):
        n, o, h, l, c = self.n, self.o, self.h, self.l, self.c
        pc = np.concatenate([[c[0]], c[:-1]])
        tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
        cs = np.cumsum(tr)
        atr = np.zeros(n)
        atr[13:] = (cs[13:] - np.concatenate([[0], cs[:-14]])) / 14.0
        self.atr = atr
        ca = np.cumsum(np.where(np.arange(n) >= 13, atr, 0.0))
        atrma = np.zeros(n)
        atrma[112:] = (ca[112:] - ca[12:n - 100]) / 100.0
        self.atrma = atrma
        self.e20, self.e50, self.e200 = (_ema(c, 2.0 / (k + 1)) for k in (20, 50, 200))
        self.rsi, self.rsif = _rsi(c, 14.0), _rsi(c, 5.0)
        self.m20, self.s20, self.hh, self.ll, self.er = _win(c, h, l)

    def compute_day(self):
        n = self.n
        self.dayOpen, self.dayHigh, self.dayLow = np.zeros(n), np.zeros(n), np.zeros(n)
        dk = self.t // 86400
        for i in range(n):
            if i == 0 or dk[i] != dk[i - 1]:
                self.dayOpen[i], self.dayHigh[i], self.dayLow[i] = self.o[i], self.h[i], self.l[i]
            else:
                self.dayOpen[i] = self.dayOpen[i - 1]
                self.dayHigh[i] = max(self.dayHigh[i - 1], self.h[i])
                self.dayLow[i] = min(self.dayLow[i - 1], self.l[i])

    def block(self, j):
        if j < 24 or j >= self.n or self.atr[j] <= 0:
            return None
        a, c = self.atr[j], self.c
        v = np.zeros(19)
        v[0:5] = [(c[j] - c[j - k]) / a for k in (1, 3, 6, 12, 24)]
        v[5], v[6], v[7] = (c[j] - self.e20[j]) / a, (c[j] - self.e50[j]) / a, (c[j] - self.e200[j]) / a
        v[8] = (self.e50[j] - self.e50[j - 5]) / a
        v[9], v[10] = self.rsi[j], self.rsif[j]
        v[11] = (c[j] - self.m20[j]) / max(self.s20[j], 1e-9)
        v[12] = 4.0 * self.s20[j] / a
        v[13] = (c[j] - self.ll[j]) / max(self.hh[j] - self.ll[j], 1e-9)
        v[14] = self.er[j]
        v[15] = a / self.atrma[j] if self.atrma[j] > 0 else 1.0
        rng = max(self.h[j] - self.l[j], 1e-9)
        o = self.o[j]
        v[16] = (c[j] - o) / rng
        v[17] = (self.h[j] - max(o, c[j])) / rng
        v[18] = (min(o, c[j]) - self.l[j]) / rng
        return v

    def last_closed(self, e):
        return int(np.searchsorted(self.t + self.dur, e, side="right")) - 1


@njit(cache=True)
def _ema(c, a):
    y = np.empty_like(c)
    y[0] = c[0]
    for i in range(1, len(c)):
        y[i] = a * c[i] + (1 - a) * y[i - 1]
    return y


@njit(cache=True)
def _rsi(c, n):
    out = np.empty_like(c)
    au = ad = 0.0
    for i in range(len(c)):
        d = 0.0 if i == 0 else c[i] - c[i - 1]
        up = d if d > 0 else 0.0
        dn = -d if d < 0 else 0.0
        if i == 0:
            au, ad = up, dn
        else:
            au = up / n + (1 - 1 / n) * au
            ad = dn / n + (1 - 1 / n) * ad
        out[i] = 100 - 100 / (1 + au / max(ad, 1e-12))
    return out


@njit(cache=True)
def _win(c, h, l):
    n = len(c)
    m20, s20, hh, ll, er = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        if i < 19:
            m20[i], hh[i], ll[i] = c[i], h[i], l[i]
            continue
        s = s2 = 0.0
        mx, mn = h[i], l[i]
        for k in range(i - 19, i + 1):
            s += c[k]
            s2 += c[k] * c[k]
            mx = max(mx, h[k])
            mn = min(mn, l[k])
        m = s / 20
        m20[i], s20[i], hh[i], ll[i] = m, np.sqrt(max(s2 / 20 - m * m, 0.0)), mx, mn
        if i >= 20:
            p = 0.0
            for k in range(i - 19, i + 1):
                p += abs(c[k] - c[k - 1])
            er[i] = abs(c[i] - c[i - 20]) / p if p > 0 else 0.0
    return m20, s20, hh, ll, er


def mirror_block(v):
    v = v.copy()
    v[0:9] = -v[0:9]
    v[9], v[10] = 100 - v[9], 100 - v[10]
    v[11] = -v[11]
    v[13] = 1 - v[13]
    v[16] = -v[16]
    v[17], v[18] = v[18], v[17]
    return v


def aggregate(h1, dur):
    key = h1.t // dur
    starts = np.flatnonzero(np.concatenate([[True], key[1:] != key[:-1]]))
    ends = np.concatenate([starts[1:], [h1.n]])
    t = key[starts] * dur
    o = h1.o[starts]
    h = np.array([h1.h[a:b].max() for a, b in zip(starts, ends)])
    l = np.array([h1.l[a:b].min() for a, b in zip(starts, ends)])
    c = h1.c[ends - 1]
    return CTF(t, o, h, l, c, dur)


def bars(m1, rule, dur):
    b = F.resample(m1, rule)
    t = (b.index.values.astype("datetime64[s]").astype(np.int64))
    return CTF(t, *(b[k].values.astype(float) for k in ("open", "high", "low", "close")), dur)


def make_features(m15, h1, h4, d1, i, d):
    e = m15.t[i] + 900
    jh, j4, jd = h1.last_closed(e), h4.last_closed(e), d1.last_closed(e)
    if i < 112 or jh < 112 or j4 < 24 or jd < 24:
        return None
    vs, vh, v4, vd = m15.block(i), h1.block(jh), h4.block(j4), d1.block(jd)
    if any(v is None for v in (vs, vh, v4, vd)):
        return None
    c, ha, da, sa = m15.c[i], h1.atr[jh], d1.atr[jd], m15.atr[i]
    today = (c - m15.dayOpen[i]) / da
    e2050 = (d1.e20[jd] - d1.e50[jd]) / da
    pdh, pdl = (c - d1.h[jd]) / ha, (c - d1.l[jd]) / ha
    dhi, dlo = (m15.dayHigh[i] - c) / ha, (c - m15.dayLow[i]) / ha
    if d < 0:
        vs, vh, v4, vd = (mirror_block(v) for v in (vs, vh, v4, vd))
        today, e2050 = -today, -e2050
        pdh, pdl = -pdl, -pdh
        dhi, dlo = dlo, dhi
    tp = 13.5
    ts = pd.Timestamp(m15.t[i], unit="s")
    x = list(vs) + list(vh) + [v4[0], v4[1], v4[5], v4[6], v4[9], v4[14], v4[13], v4[8]] + \
        [vd[0], vd[1], vd[3], vd[5], vd[6], vd[8], vd[9], vd[14], vd[13], e2050, today] + \
        [pdh, pdl, dhi, dlo, tp / ha, 30.0 / ha, sa / ha, ts.hour + ts.minute / 60.0, ts.weekday(), d]
    return np.array(x)


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]           # comparison window
    m1 = E.load_m1()
    # emulate the broker windows like the EA's training call (H1 starts 400 days earlier)
    m1w = m1[(m1.stime >= pd.Timestamp(a) - pd.Timedelta(days=400)) & (m1.stime < pd.Timestamp(b))]
    m15 = bars(m1w[m1w.stime >= pd.Timestamp(a) - pd.Timedelta(days=30)], "15min", 900)
    h1 = bars(m1w, "1h", 3600)
    h4, d1 = aggregate(h1, 14400), aggregate(h1, 86400)
    for tf in (m15, h1, h4, d1):
        tf.compute()
    m15.compute_day()
    D = load("cand_t5_4")
    D = D[(D.index >= a) & (D.index < b)]
    feats = [k for k in D.columns if k not in ["ent", "ex", "pnl", "reason", "mfe", "mae", "hold", "close", "res", "valid"]]
    t15 = m15.t
    rows = []
    for ts, r in D.sample(min(3000, len(D)), random_state=1).iterrows():
        i = int(np.searchsorted(t15, int(ts.value // 10 ** 9)))
        if i >= m15.n or t15[i] != ts.value // 10 ** 9:
            continue
        x = make_features(m15, h1, h4, d1, i, int(r.dir))
        if x is None:
            continue
        rows.append(np.abs(x - r[feats].values.astype(float)))
    A = np.array(rows)
    err = pd.DataFrame(A, columns=feats)
    print("compared", len(A), "candidates")
    q = err.quantile(0.99).sort_values(ascending=False)
    print("99th pct abs diff (worst 15):")
    print(q.head(15).round(5).to_string())
    print("median abs diff max over features: %.2e" % err.median().max())
