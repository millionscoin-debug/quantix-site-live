//+------------------------------------------------------------------+
//|                                                CortexML_AI.mq5   |
//|        Cortex ML AI - XAUUSD intraday-momentum engine with an    |
//|        online-learning logistic model and Adaptive Profit Shield |
//|                                         Quantix Engineering 2026 |
//+------------------------------------------------------------------+
#property copyright   "Quantix Engineering"
#property link        "https://quantixengine.com"
#property version     "1.00"
#property description "Cortex ML AI: trades XAUUSD in the direction of strong intraday moves."
#property description "An online logistic model (re-trained every month on the broker's own history"
#property description "plus the EA's real results) vetoes weak core signals and adds confident ones."
#property description "Hard SL $30, TP $12-15, Adaptive Profit Shield trailing that never locks less than $3."

#include <Trade\Trade.mqh>

//--- number of model features (must match the research pipeline: research/features.py)
#define NF 67

enum ENUM_LOT_MODE
  {
   LOT_FIXED        = 0,   // Fixed lot
   LOT_PER_BALANCE  = 1,   // 0.01 lot per N of balance
   LOT_RISK_PERCENT = 2    // % of balance lost at the hard stop
  };

enum ENUM_TP_MODE
  {
   TP_FIXED    = 0,        // Fixed take profit
   TP_ADAPTIVE = 1         // Adaptive 12..15 by H1 volatility
  };

enum ENUM_TRAIL_PROFILE
  {
   TRAIL_SHIELD   = 0,     // Shield: start +5, lock +4, distance 3 (researched default)
   TRAIL_BALANCED = 1,     // Balanced: start +6, lock +4, distance 3
   TRAIL_RUNNER   = 2,     // Runner: start +8, lock +5, distance 3
   TRAIL_CUSTOM   = 3,     // Custom values below
   TRAIL_OFF      = 4      // Off (pure TP / SL)
  };

//--- inputs
input group "=== Money management ==="
input ENUM_LOT_MODE      InpLotMode          = LOT_FIXED;   // Lot mode
input double             InpFixedLot         = 0.01;        // Fixed lot
input double             InpBalancePer001    = 1000.0;      // Balance per 0.01 lot (per-balance mode)
input double             InpRiskPercent      = 1.0;         // Risk % at hard stop (risk mode)
input double             InpMaxLot           = 5.0;         // Maximum lot

input group "=== Exits (USD of XAUUSD price = USD per 0.01 lot) ==="
input double             InpStopLossUSD      = 30.0;        // Hard stop loss
input ENUM_TP_MODE       InpTPMode           = TP_FIXED;    // Take profit mode
input double             InpTakeProfitUSD    = 13.5;        // Fixed take profit
input double             InpTPMinUSD         = 12.0;        // Adaptive TP minimum
input double             InpTPMaxUSD         = 15.0;        // Adaptive TP maximum

input group "=== Adaptive Profit Shield (trailing) ==="
input ENUM_TRAIL_PROFILE InpTrailProfile     = TRAIL_SHIELD;// Trailing profile
input double             InpTrailStartUSD    = 5.0;         // Custom: activation profit
input double             InpTrailLockUSD     = 4.0;         // Custom: profit locked at activation (>= 3)
input double             InpTrailDistUSD     = 3.0;         // Custom: distance from the peak
input double             InpTightenAt        = 0.8;         // Tighten when profit >= this share of TP
input double             InpTightenK         = 0.6;         // Distance multiplier near TP
input double             InpTrailStepUSD     = 0.5;         // Minimal SL improvement per modification

input group "=== Signal engine: intraday momentum ==="
input double             InpCoreDayMove      = 1.0;         // Core signal: day move >= x daily ATR
input double             InpMinDayMove       = 0.6;         // ML tier: day move >= x daily ATR
input int                InpHourFrom         = 8;           // First entry hour (NY+7 server clock)
input int                InpHourTo           = 23;          // Entries until hour (exclusive)
input int                InpMaxPositions     = 1;           // Max simultaneous positions
input int                InpMinMinutesGap    = 0;           // Min minutes between entries (0 = off)
input int                InpFridayLastHour   = 23;          // No new entries on Friday from hour

input group "=== Machine learning core ==="
input bool               InpUseML            = true;        // Use the ML model
input double             InpTrainYears       = 3.0;         // Rolling training window (years)
input int                InpMinSamples       = 300;         // Min labelled samples to activate the model
input double             InpRegC             = 0.05;        // Inverse L2 strength (C)
input double             InpCoreVeto         = 0.03;        // Core veto: skip if p < base - veto
input double             InpMLMargin         = 0.04;        // ML tier: take if p > base + margin
input double             InpLabelSpreadUSD   = 0.25;        // Spread assumed when labelling history
input double             InpCommissionPerLot = 7.0;         // Round-turn commission per 1.0 lot
input bool               InpLearnFromTrades  = true;        // Feed real trade results back into the data
input bool               InpPersistMemory    = true;        // Live: keep samples/feedback in Common files

input group "=== Protection ==="
input double             InpMaxSpreadUSD     = 0.80;        // Max spread at entry
input double             InpDailyLossPct     = 0.0;         // Daily loss limit % (0 = off)
input double             InpEquityStopPct    = 0.0;         // Stop trading below this % drawdown (0 = off)

input group "=== Server clock ==="
input int                InpServerGMTWinter  = 2;           // Broker GMT offset in winter
input bool               InpServerUsDst      = true;        // Broker follows US daylight saving

input group "=== General ==="
input long               InpMagic            = 20261024;    // Magic number
input string             InpComment          = "CortexML";  // Order comment
input datetime           InpTradeStart       = D'2000.01.01';// No trading before (tester warm-up)
input bool               InpShowPanel        = true;        // Show info panel
input bool               InpVerbose          = false;       // Verbose log

//--- forward declarations
bool Train();
void SaveMemory();
void LoadMemory();

//--- trade-management constants derived from the inputs
double g_act, g_lock, g_dist;
bool   g_trailOn;

//+------------------------------------------------------------------+
//| Time helpers: everything is computed on the "NY+7" clock         |
//| (UTC+2 winter / UTC+3 US summer), the clock of IC Markets & co.  |
//+------------------------------------------------------------------+
int      g_dstYear = -1;
datetime g_dstFrom = 0, g_dstTo = 0;

bool IsUsDst(datetime t)
  {
   MqlDateTime s;
   TimeToStruct(t, s);
   if(s.year != g_dstYear)
     {
      MqlDateTime m;
      datetime mar = StringToTime(IntegerToString(s.year) + ".03.08");
      TimeToStruct(mar, m);
      g_dstFrom = mar + ((7 - m.day_of_week) % 7) * 86400 + 2 * 3600;
      datetime nov = StringToTime(IntegerToString(s.year) + ".11.01");
      TimeToStruct(nov, m);
      g_dstTo   = nov + ((7 - m.day_of_week) % 7) * 86400 + 2 * 3600;
      g_dstYear = s.year;
     }
   return(t >= g_dstFrom && t < g_dstTo);
  }

bool g_shiftZero = true;

datetime ToNY7(datetime serverTime)
  {
   if(g_shiftZero)
      return(serverTime);
   bool dst = IsUsDst(serverTime);
   int srv  = InpServerGMTWinter + ((InpServerUsDst && dst) ? 1 : 0);
   int ny7  = 2 + (dst ? 1 : 0);
   return(serverTime + (ny7 - srv) * 3600);
  }

long DayKey(datetime t) { return((long)t / 86400); }

//+------------------------------------------------------------------+
//| Bar series with the indicator set of research/features.tf_block  |
//+------------------------------------------------------------------+
class CTF
  {
public:
   int               n;
   int               dur;
   datetime          t[];
   double            o[], h[], l[], c[];
   double            atr[], e20[], e50[], e200[], rsi[], rsif[], m20[], s20[], hh[], ll[], er[], atrma[];
   //--- intraday helpers (used by the signal timeframe only)
   double            dayOpen[], dayHigh[], dayLow[];

   void              Reset(int durSec)
     {
      n = 0;
      dur = durSec;
      ArrayResize(t, 0, 100000);
      ArrayResize(o, 0, 100000);
      ArrayResize(h, 0, 100000);
      ArrayResize(l, 0, 100000);
      ArrayResize(c, 0, 100000);
     }

   void              Add(datetime tt, double oo, double hi, double lo, double cc)
     {
      ArrayResize(t, n + 1, 100000);
      ArrayResize(o, n + 1, 100000);
      ArrayResize(h, n + 1, 100000);
      ArrayResize(l, n + 1, 100000);
      ArrayResize(c, n + 1, 100000);
      t[n] = tt;
      o[n] = oo;
      h[n] = hi;
      l[n] = lo;
      c[n] = cc;
      n++;
     }

   //--- merge a bar into the last one (used when aggregating H1 -> H4 / D1)
   void              Merge(double hi, double lo, double cc)
     {
      if(hi > h[n - 1])
         h[n - 1] = hi;
      if(lo < l[n - 1])
         l[n - 1] = lo;
      c[n - 1] = cc;
     }

   void              Compute()
     {
      ArrayResize(atr, n);
      ArrayResize(e20, n);
      ArrayResize(e50, n);
      ArrayResize(e200, n);
      ArrayResize(rsi, n);
      ArrayResize(rsif, n);
      ArrayResize(m20, n);
      ArrayResize(s20, n);
      ArrayResize(hh, n);
      ArrayResize(ll, n);
      ArrayResize(er, n);
      ArrayResize(atrma, n);
      if(n == 0)
         return;
      //--- ATR(14) = SMA of true range
      double sumTR = 0.0;
      double tr[];
      ArrayResize(tr, n);
      for(int i = 0; i < n; i++)
        {
         double pc = (i == 0) ? c[0] : c[i - 1];
         tr[i] = MathMax(h[i] - l[i], MathMax(MathAbs(h[i] - pc), MathAbs(l[i] - pc)));
         sumTR += tr[i];
         if(i >= 14)
            sumTR -= tr[i - 14];
         atr[i] = (i >= 13) ? sumTR / 14.0 : 0.0;
        }
      //--- ATR moving average (100)
      double sumA = 0.0;
      for(int i = 0; i < n; i++)
        {
         if(i >= 13)
            sumA += atr[i];
         if(i >= 113)
            sumA -= atr[i - 100];
         atrma[i] = (i >= 112) ? sumA / 100.0 : 0.0;
        }
      //--- EMAs (pandas ewm span, adjust=False)
      double a20 = 2.0 / 21.0, a50 = 2.0 / 51.0, a200 = 2.0 / 201.0;
      e20[0] = c[0];
      e50[0] = c[0];
      e200[0] = c[0];
      for(int i = 1; i < n; i++)
        {
         e20[i]  = a20  * c[i] + (1.0 - a20)  * e20[i - 1];
         e50[i]  = a50  * c[i] + (1.0 - a50)  * e50[i - 1];
         e200[i] = a200 * c[i] + (1.0 - a200) * e200[i - 1];
        }
      //--- RSI 14 and 5, Wilder smoothing
      double au14 = 0, ad14 = 0, au5 = 0, ad5 = 0;
      for(int i = 0; i < n; i++)
        {
         double d  = (i == 0) ? 0.0 : c[i] - c[i - 1];
         double up = (d > 0) ? d : 0.0;
         double dn = (d < 0) ? -d : 0.0;
         if(i == 0)
           {
            au14 = up;
            ad14 = dn;
            au5 = up;
            ad5 = dn;
           }
         else
           {
            au14 = up / 14.0 + (1.0 - 1.0 / 14.0) * au14;
            ad14 = dn / 14.0 + (1.0 - 1.0 / 14.0) * ad14;
            au5  = up / 5.0  + (1.0 - 1.0 / 5.0)  * au5;
            ad5  = dn / 5.0  + (1.0 - 1.0 / 5.0)  * ad5;
           }
         rsi[i]  = 100.0 - 100.0 / (1.0 + au14 / MathMax(ad14, 1e-12));
         rsif[i] = 100.0 - 100.0 / (1.0 + au5  / MathMax(ad5,  1e-12));
        }
      //--- 20-bar windows: mean, population std, highest high, lowest low, efficiency ratio
      for(int i = 0; i < n; i++)
        {
         if(i < 19)
           {
            m20[i] = c[i];
            s20[i] = 0;
            hh[i] = h[i];
            ll[i] = l[i];
            er[i] = 0;
            continue;
           }
         double s = 0, s2 = 0, mx = h[i], mn = l[i];
         for(int k = i - 19; k <= i; k++)
           {
            s  += c[k];
            s2 += c[k] * c[k];
            if(h[k] > mx)
               mx = h[k];
            if(l[k] < mn)
               mn = l[k];
           }
         double m = s / 20.0;
         m20[i] = m;
         s20[i] = MathSqrt(MathMax(s2 / 20.0 - m * m, 0.0));
         hh[i] = mx;
         ll[i] = mn;
         if(i >= 20)
           {
            double path = 0;
            for(int k = i - 19; k <= i; k++)
               path += MathAbs(c[k] - c[k - 1]);
            er[i] = (path > 0) ? MathAbs(c[i] - c[i - 20]) / path : 0.0;
           }
         else
            er[i] = 0;
        }
     }

   //--- server-day open / running high / running low for every bar
   void              ComputeDay()
     {
      ArrayResize(dayOpen, n);
      ArrayResize(dayHigh, n);
      ArrayResize(dayLow, n);
      for(int i = 0; i < n; i++)
        {
         if(i == 0 || DayKey(t[i]) != DayKey(t[i - 1]))
           {
            dayOpen[i] = o[i];
            dayHigh[i] = h[i];
            dayLow[i]  = l[i];
           }
         else
           {
            dayOpen[i] = dayOpen[i - 1];
            dayHigh[i] = MathMax(dayHigh[i - 1], h[i]);
            dayLow[i]  = MathMin(dayLow[i - 1], l[i]);
           }
        }
     }

   //--- 19 raw block values at bar j (order of research/features.tf_block)
   bool              Block(int j, double &v[])
     {
      if(j < 24 || j >= n)
         return(false);
      double a = atr[j];
      if(a <= 0)
         return(false);
      v[0]  = (c[j] - c[j - 1])  / a;
      v[1]  = (c[j] - c[j - 3])  / a;
      v[2]  = (c[j] - c[j - 6])  / a;
      v[3]  = (c[j] - c[j - 12]) / a;
      v[4]  = (c[j] - c[j - 24]) / a;
      v[5]  = (c[j] - e20[j])  / a;
      v[6]  = (c[j] - e50[j])  / a;
      v[7]  = (c[j] - e200[j]) / a;
      v[8]  = (e50[j] - e50[j - 5]) / a;
      v[9]  = rsi[j];
      v[10] = rsif[j];
      v[11] = (c[j] - m20[j]) / MathMax(s20[j], 1e-9);
      v[12] = 4.0 * s20[j] / a;
      v[13] = (c[j] - ll[j]) / MathMax(hh[j] - ll[j], 1e-9);
      v[14] = er[j];
      v[15] = (atrma[j] > 0) ? a / atrma[j] : 1.0;
      double rng = MathMax(h[j] - l[j], 1e-9);
      v[16] = (c[j] - o[j]) / rng;
      v[17] = (h[j] - MathMax(o[j], c[j])) / rng;
      v[18] = (MathMin(o[j], c[j]) - l[j]) / rng;
      return(true);
     }

   //--- last bar fully closed at time e
   int               LastClosed(datetime e)
     {
      int lo = 0, hi = n - 1, ans = -1;
      while(lo <= hi)
        {
         int mid = (lo + hi) / 2;
         if(t[mid] + dur <= e)
           {
            ans = mid;
            lo = mid + 1;
           }
         else
            hi = mid - 1;
        }
      return(ans);
     }

   //--- first bar starting at or after time e
   int               FirstAtOrAfter(datetime e)
     {
      int lo = 0, hi = n - 1, ans = -1;
      while(lo <= hi)
        {
         int mid = (lo + hi) / 2;
         if(t[mid] >= e)
           {
            ans = mid;
            hi = mid - 1;
           }
         else
            lo = mid + 1;
        }
      return(ans);
     }
  };

//+------------------------------------------------------------------+
//| Data loading                                                     |
//+------------------------------------------------------------------+
bool LoadTF(ENUM_TIMEFRAMES tf, datetime from, datetime to, CTF &dst, int durSec)
  {
   MqlRates r[];
   ArraySetAsSeries(r, false);
   int got = CopyRates(_Symbol, tf, from, to, r);
   dst.Reset(durSec);
   if(got <= 0)
      return(false);
   for(int i = 0; i < got; i++)
      dst.Add(ToNY7(r[i].time), r[i].open, r[i].high, r[i].low, r[i].close);
   return(true);
  }

//--- aggregate (shifted) H1 bars into H4 and D1 exactly like pandas resample on the NY+7 clock
void Aggregate(CTF &h1, CTF &dst, int durSec)
  {
   dst.Reset(durSec);
   long last = -1;
   for(int i = 0; i < h1.n; i++)
     {
      long key = (long)h1.t[i] / durSec;
      if(key != last)
        {
         dst.Add((datetime)(key * durSec), h1.o[i], h1.h[i], h1.l[i], h1.c[i]);
         last = key;
        }
      else
         dst.Merge(h1.h[i], h1.l[i], h1.c[i]);
     }
  }

//--- the four timeframes used by the model
CTF g_m15, g_h1, g_h4, g_d1;   // live window
CTF t_m15, t_h1, t_h4, t_d1;   // training history
CTF t_path;                    // M1 (+ M5 fallback) path for labelling

bool LoadSet(datetime from, datetime to, CTF &m15, CTF &h1, CTF &h4, CTF &d1)
  {
   if(!LoadTF(PERIOD_M15, from, to, m15, 900))
      return(false);
   //--- the daily block needs ~280 trading days before the first signal bar
   if(!LoadTF(PERIOD_H1, from - 400 * 86400, to, h1, 3600))
      return(false);
   Aggregate(h1, h4, 14400);
   Aggregate(h1, d1, 86400);
   m15.Compute();
   m15.ComputeDay();
   h1.Compute();
   h4.Compute();
   d1.Compute();
   return(m15.n > 200 && h1.n > 200 && d1.n > 30);
  }

//--- M1 path for labelling; older periods without M1 fall back to M5 bars
bool LoadPath(datetime from, datetime to)
  {
   CTF m1, m5;
   bool ok1 = LoadTF(PERIOD_M1, from, to, m1, 60);
   datetime m1Start = (ok1 && m1.n > 0) ? m1.t[0] : to;
   t_path.Reset(60);
   if(m1Start > from + 86400)
     {
      if(LoadTF(PERIOD_M5, from, to, m5, 300))
         for(int i = 0; i < m5.n && m5.t[i] < m1Start; i++)
            t_path.Add(m5.t[i], m5.o[i], m5.h[i], m5.l[i], m5.c[i]);
     }
   if(ok1)
      for(int i = 0; i < m1.n; i++)
         t_path.Add(m1.t[i], m1.o[i], m1.h[i], m1.l[i], m1.c[i]);
   return(t_path.n > 1000);
  }

//+------------------------------------------------------------------+
//| Feature vector (exactly research/features.py + mirror())         |
//+------------------------------------------------------------------+
void MirrorBlock(double &v[])
  {
   for(int k = 0; k <= 8; k++)
      v[k] = -v[k];             // returns, EMA distances, slope
   v[9]  = 100.0 - v[9];        // rsi
   v[10] = 100.0 - v[10];       // fast rsi
   v[11] = -v[11];              // bollinger z
   v[13] = 1.0 - v[13];         // position in 20-bar range
   v[16] = -v[16];              // body
   double u = v[17];            // upper <-> lower wick
   v[17] = v[18];
   v[18] = u;
  }

double TPForVolatility(double h1volr)
  {
   if(InpTPMode == TP_FIXED)
      return(InpTakeProfitUSD);
   double k = MathMax(0.0, MathMin(1.0, (h1volr - 0.8) / 0.6));
   return(InpTPMinUSD + (InpTPMaxUSD - InpTPMinUSD) * k);
  }

//--- raw day move of signal bar i in daily ATR (sign = direction of the day)
bool DayMove(CTF &m15, CTF &d1, int i, double &dm)
  {
   datetime e = m15.t[i] + 900;
   int jd = d1.LastClosed(e);
   if(jd < 20 || d1.atr[jd] <= 0)
      return(false);
   dm = (m15.c[i] - m15.dayOpen[i]) / d1.atr[jd];
   return(true);
  }

double ServerHour(datetime t)
  {
   MqlDateTime s;
   TimeToStruct(t, s);
   return(s.hour + s.min / 60.0);
  }

bool MakeFeatures(CTF &m15, CTF &h1, CTF &h4, CTF &d1, int i, int dir, double &x[], double &tpUsed)
  {
   if(i < 112)
      return(false);
   datetime e = m15.t[i] + 900;
   int jh = h1.LastClosed(e);
   int j4 = h4.LastClosed(e);
   int jd = d1.LastClosed(e);
   if(jh < 112 || j4 < 24 || jd < 24)
      return(false);
   double vs[19], vh[19], v4[19], vd[19];
   if(!m15.Block(i, vs) || !h1.Block(jh, vh) || !h4.Block(j4, v4) || !d1.Block(jd, vd))
      return(false);
   double c  = m15.c[i];
   double ha = h1.atr[jh];
   double da = d1.atr[jd];
   double sa = m15.atr[i];
   if(ha <= 0 || da <= 0 || sa <= 0)
      return(false);
   double today = (c - m15.dayOpen[i]) / da;
   double e2050 = (d1.e20[jd] - d1.e50[jd]) / da;
   double pdh = (c - d1.h[jd]) / ha;
   double pdl = (c - d1.l[jd]) / ha;
   double dhi = (m15.dayHigh[i] - c) / ha;
   double dlo = (c - m15.dayLow[i]) / ha;
   if(dir < 0)
     {
      MirrorBlock(vs);
      MirrorBlock(vh);
      MirrorBlock(v4);
      MirrorBlock(vd);
      today = -today;
      e2050 = -e2050;
      double a = pdh;
      pdh = -pdl;
      pdl = -a;
      a = dhi;
      dhi = dlo;
      dlo = a;
     }
   tpUsed = TPForVolatility(vh[15]);              // H1 ATR / its 100-bar mean (never mirrored)
   int p = 0;
   for(int k = 0; k < 19; k++)
      x[p++] = vs[k];
   for(int k = 0; k < 19; k++)
      x[p++] = vh[k];
   x[p++] = v4[0];
   x[p++] = v4[1];
   x[p++] = v4[5];
   x[p++] = v4[6];
   x[p++] = v4[9];
   x[p++] = v4[14];
   x[p++] = v4[13];
   x[p++] = v4[8];
   x[p++] = vd[0];
   x[p++] = vd[1];
   x[p++] = vd[3];
   x[p++] = vd[5];
   x[p++] = vd[6];
   x[p++] = vd[8];
   x[p++] = vd[9];
   x[p++] = vd[14];
   x[p++] = vd[13];
   x[p++] = e2050;
   x[p++] = today;
   x[p++] = pdh;
   x[p++] = pdl;
   x[p++] = dhi;
   x[p++] = dlo;
   x[p++] = tpUsed / ha;
   x[p++] = InpStopLossUSD / ha;
   x[p++] = sa / ha;
   x[p++] = ServerHour(m15.t[i]);
   MqlDateTime s;
   TimeToStruct(m15.t[i], s);
   x[p++] = (s.day_of_week + 6) % 7;     // Monday = 0 like pandas
   x[p++] = dir;
   return(p == NF);
  }

//+------------------------------------------------------------------+
//| Exact trade simulator (port of research/engine.sim_trade)        |
//+------------------------------------------------------------------+
struct SimResult
  {
   double            pnl;
   int               exitIdx;
   int               reason;   // 1 TP, 2 hard SL, 3 shield SL, 5 data end
  };

bool Leg(int dir, double p0, double p1, double entry, double &slv, double tpv, double &peak,
         double tpDist, double &px)
  {
   if(dir > 0)
     {
      if(p1 >= p0)
        {
         if(p1 >= tpv)
           {
            px = tpv;
            return(true);
           }
         if(g_trailOn && p1 > peak)
           {
            peak = p1;
            double prof = peak - entry;
            if(prof >= g_act)
              {
               double d = g_dist;
               if(prof >= InpTightenAt * tpDist)
                  d = g_dist * InpTightenK;
               double ns = MathMax(peak - d, entry + g_lock);
               if(ns >= slv + InpTrailStepUSD)
                  slv = ns;
              }
           }
         return(false);
        }
      if(p1 <= slv)
        {
         px = MathMin(slv, p0);
         return(true);
        }
      return(false);
     }
   if(p1 <= p0)
     {
      if(p1 <= tpv)
        {
         px = tpv;
         return(true);
        }
      if(g_trailOn && p1 < peak)
        {
         peak = p1;
         double prof = entry - peak;
         if(prof >= g_act)
           {
            double d = g_dist;
            if(prof >= InpTightenAt * tpDist)
               d = g_dist * InpTightenK;
            double ns = MathMin(peak + d, entry - g_lock);
            if(ns <= slv - InpTrailStepUSD)
               slv = ns;
           }
        }
      return(false);
     }
   if(p1 >= slv)
     {
      px = MathMax(slv, p0);
      return(true);
     }
   return(false);
  }

SimResult SimTrade(CTF &path, int i0, int dir, double tpDist)
  {
   SimResult r;
   double spr  = InpLabelSpreadUSD;
   double comm = InpCommissionPerLot / 100.0;      // per 0.01 lot
   double q = (dir > 0) ? 0.0 : spr;               // exit-side quote offset
   double entry = path.o[i0] + ((dir > 0) ? spr : 0.0);
   double slv = entry - dir * InpStopLossUSD;
   double hard = slv;
   double tpv = entry + dir * tpDist;
   double peak = entry;
   for(int i = i0; i < path.n; i++)
     {
      double oo = path.o[i] + q, hh = path.h[i] + q, ll = path.l[i] + q, cc = path.c[i] + q;
      if(i > i0)
        {
         if((dir > 0 && oo <= slv) || (dir < 0 && oo >= slv))
           {
            r.pnl = dir * (oo - entry) - comm;
            r.exitIdx = i;
            r.reason = (slv == hard) ? 2 : 3;
            return(r);
           }
         if((dir > 0 && oo >= tpv) || (dir < 0 && oo <= tpv))
           {
            r.pnl = dir * (oo - entry) - comm;
            r.exitIdx = i;
            r.reason = 1;
            return(r);
           }
        }
      double a1 = (cc >= oo) ? ll : hh;
      double a2 = (cc >= oo) ? hh : ll;
      double pts[4];
      pts[0] = oo;
      pts[1] = a1;
      pts[2] = a2;
      pts[3] = cc;
      for(int k = 0; k < 3; k++)
        {
         double px = 0;
         if(Leg(dir, pts[k], pts[k + 1], entry, slv, tpv, peak, tpDist, px))
           {
            r.pnl = dir * (px - entry) - comm;
            r.exitIdx = i;
            r.reason = (px == tpv) ? 1 : ((slv == hard) ? 2 : 3);
            return(r);
           }
        }
     }
   r.pnl = dir * (path.c[path.n - 1] + q - entry) - comm;
   r.exitIdx = path.n - 1;
   r.reason = 5;
   return(r);
  }

//+------------------------------------------------------------------+
//| Labelled sample memory                                           |
//+------------------------------------------------------------------+
struct Sample
  {
   datetime          t;        // signal bar start (NY+7 clock)
   datetime          te;       // exit time of the (simulated or real) trade
   int               dir;
   double            dm;       // day move in the trade direction (daily ATR)
   double            pnl;      // $ per 0.01 lot
   int               real;     // 1 = label comes from a real trade of this EA
   double            x[NF];
  };

struct Feedback
  {
   datetime          t;
   int               dir;
   double            pnl;
   datetime          te;
  };

Sample   g_S[];
int      g_nS = 0;
Feedback g_F[];
int      g_nF = 0;

//--- model
double   g_mu[NF], g_sd[NF], g_w[NF];
double   g_b = 0, g_base = 0;
bool     g_modelOk = false;
int      g_trainN = 0;
datetime g_trainedAt = 0;
int      g_trainMonth = -1;
double   g_trainAcc = 0;

//--- runtime
datetime g_lastBar = 0;
datetime g_lastEntry = 0;
double   g_lastP = -1, g_lastE = 0, g_lastDM = 0;
int      g_lastDir = 0;
string   g_lastDecision = "waiting for the first closed bar";
double   g_dayStartBalance = 0;
int      g_dayKey = -1;
double   g_peakEquity = 0;

ulong    g_posTicket[];
double   g_posPeak[];
datetime g_posSigT[];
int      g_posSigDir[];

CTrade   trade;

//+------------------------------------------------------------------+
//| Logistic regression with L2 (Newton / IRLS), = sklearn lbfgs fit |
//+------------------------------------------------------------------+
bool Cholesky(double &a[], int d)
  {
   for(int j = 0; j < d; j++)
     {
      double s = a[j * d + j];
      for(int k = 0; k < j; k++)
         s -= a[j * d + k] * a[j * d + k];
      if(s <= 1e-12)
         return(false);
      double ljj = MathSqrt(s);
      a[j * d + j] = ljj;
      for(int i = j + 1; i < d; i++)
        {
         double v = a[i * d + j];
         for(int k = 0; k < j; k++)
            v -= a[i * d + k] * a[j * d + k];
         a[i * d + j] = v / ljj;
        }
     }
   return(true);
  }

void CholSolve(const double &L[], int d, const double &g[], double &x[])
  {
   double y[];
   ArrayResize(y, d);
   for(int i = 0; i < d; i++)
     {
      double v = g[i];
      for(int k = 0; k < i; k++)
         v -= L[i * d + k] * y[k];
      y[i] = v / L[i * d + i];
     }
   for(int i = d - 1; i >= 0; i--)
     {
      double v = y[i];
      for(int k = i + 1; k < d; k++)
         v -= L[k * d + i] * x[k];
      x[i] = v / L[i * d + i];
     }
  }

double Sigmoid(double z)
  {
   if(z > 35)
      return(1.0);
   if(z < -35)
      return(0.0);
   return(1.0 / (1.0 + MathExp(-z)));
  }

bool FitLogistic(const double &Z[], const double &Y[], int N)
  {
   int d = NF + 1;
   double lambda = 1.0 / MathMax(InpRegC, 1e-6);
   double W[];
   ArrayResize(W, d);
   ArrayInitialize(W, 0.0);
   double ybar = 0;
   for(int i = 0; i < N; i++)
      ybar += Y[i];
   ybar /= N;
   ybar = MathMin(MathMax(ybar, 0.01), 0.99);
   W[NF] = MathLog(ybar / (1.0 - ybar));
   double G[], H[], step[];
   ArrayResize(G, d);
   ArrayResize(H, d * d);
   ArrayResize(step, d);
   for(int it = 0; it < 30; it++)
     {
      ArrayInitialize(G, 0.0);
      ArrayInitialize(H, 0.0);
      for(int i = 0; i < N; i++)
        {
         int off = i * NF;
         double eta = W[NF];
         for(int k = 0; k < NF; k++)
            eta += W[k] * Z[off + k];
         double p = Sigmoid(eta);
         double r = p - Y[i];
         double w = MathMax(p * (1.0 - p), 1e-9);
         for(int k = 0; k < NF; k++)
           {
            double zk = Z[off + k];
            G[k] += r * zk;
            double wz = w * zk;
            int row = k * d;
            for(int m = 0; m <= k; m++)
               H[row + m] += wz * Z[off + m];
            H[row + NF] += wz;
           }
         G[NF] += r;
         H[NF * d + NF] += w;
        }
      for(int k = 0; k < NF; k++)
        {
         G[k] += lambda * W[k];
         H[k * d + k] += lambda;
        }
      H[NF * d + NF] += 1e-6;
      //--- symmetrise (lower triangle filled above, bias column stored in row k)
      for(int k = 0; k < NF; k++)
        {
         for(int m = 0; m < k; m++)
            H[m * d + k] = H[k * d + m];
         H[NF * d + k] = H[k * d + NF];
        }
      if(!Cholesky(H, d))
         return(false);
      CholSolve(H, d, G, step);
      double mx = 0;
      for(int k = 0; k < d; k++)
        {
         W[k] -= step[k];
         mx = MathMax(mx, MathAbs(step[k]));
        }
      if(mx < 1e-7)
         break;
     }
   for(int k = 0; k < NF; k++)
      g_w[k] = W[k];
   g_b = W[NF];
   return(true);
  }

double Predict(const double &x[])
  {
   if(!g_modelOk)
      return(-1);
   double eta = g_b;
   for(int k = 0; k < NF; k++)
     {
      double z = (x[k] - g_mu[k]) / g_sd[k];
      z = MathMax(-4.0, MathMin(4.0, z));
      eta += g_w[k] * z;
     }
   return(Sigmoid(eta));
  }

//+------------------------------------------------------------------+
//| Build the labelled dataset from history and train                |
//+------------------------------------------------------------------+
int FindSample(datetime t, int dir)
  {
   for(int i = g_nS - 1; i >= 0; i--)
      if(g_S[i].t == t && g_S[i].dir == dir)
         return(i);
   return(-1);
  }

void ApplyFeedback()
  {
   for(int f = 0; f < g_nF; f++)
     {
      int k = FindSample(g_F[f].t, g_F[f].dir);
      if(k >= 0)
        {
         g_S[k].pnl = g_F[f].pnl;
         g_S[k].te = g_F[f].te;
         g_S[k].real = 1;
        }
     }
  }

int BuildFromHistory(datetime from, datetime to)
  {
   if(!LoadSet(from, to, t_m15, t_h1, t_h4, t_d1))
      return(-1);
   if(!LoadPath(from, to))
      return(-2);
   Sample built[];
   int nb = 0;
   ArrayResize(built, 0, 20000);
   double x[NF];
   datetime nowNY = ToNY7(TimeCurrent());
   for(int i = 112; i < t_m15.n - 1; i++)
     {
      double hr = ServerHour(t_m15.t[i]);
      if(hr < InpHourFrom || hr >= InpHourTo)
         continue;
      double dmRaw = 0;
      if(!DayMove(t_m15, t_d1, i, dmRaw))
         continue;
      int dir = (dmRaw > 0) ? 1 : -1;
      double dm = MathAbs(dmRaw);
      if(dm <= InpMinDayMove)
         continue;
      double tpUsed = InpTakeProfitUSD;
      if(!MakeFeatures(t_m15, t_h1, t_h4, t_d1, i, dir, x, tpUsed))
         continue;
      datetime e = t_m15.t[i] + 900;
      int i0 = t_path.FirstAtOrAfter(e);
      if(i0 < 0 || t_path.t[i0] - e >= 3600)
         continue;
      SimResult r = SimTrade(t_path, i0, dir, tpUsed);
      if(r.reason == 5)
         continue;                             // still open at the end of the data
      datetime te = t_path.t[r.exitIdx];
      if(te >= nowNY)
         continue;
      ArrayResize(built, nb + 1, 20000);
      built[nb].t = t_m15.t[i];
      built[nb].te = te;
      built[nb].dir = dir;
      built[nb].dm = dm;
      built[nb].pnl = r.pnl;
      built[nb].real = 0;
      for(int k = 0; k < NF; k++)
         built[nb].x[k] = x[k];
      nb++;
     }
   //--- merge: keep remembered samples older than what history could rebuild
   datetime firstBuilt = (nb > 0) ? built[0].t : ToNY7(to);
   Sample merged[];
   int nm = 0;
   ArrayResize(merged, 0, 20000);
   datetime winFrom = ToNY7(from);
   for(int i = 0; i < g_nS; i++)
     {
      if(g_S[i].t < firstBuilt && g_S[i].t >= winFrom && g_S[i].te < nowNY)
        {
         ArrayResize(merged, nm + 1, 20000);
         merged[nm++] = g_S[i];
        }
     }
   for(int i = 0; i < nb; i++)
     {
      ArrayResize(merged, nm + 1, 20000);
      merged[nm++] = built[i];
     }
   ArrayResize(g_S, nm);
   for(int i = 0; i < nm; i++)
      g_S[i] = merged[i];
   g_nS = nm;
   ApplyFeedback();
   return(nb);
  }

bool Train()
  {
   uint t0 = GetTickCount();
   datetime now = TimeCurrent();
   datetime from = (datetime)((long)now - (long)(InpTrainYears * 365.0 * 86400.0));
   int nb = BuildFromHistory(from, now);
   if(nb < 0)
     {
      if(InpVerbose)
         Print("CortexML: history not ready (", nb, ")");
      return(false);
     }
   int N = g_nS;
   MqlDateTime s;
   TimeToStruct(now, s);
   g_trainMonth = s.year * 12 + s.mon;
   if(!InpUseML || N < InpMinSamples)
     {
      g_modelOk = false;
      g_trainN = N;
      PrintFormat("CortexML: %d labelled samples (< %d) -> core rule only until enough history", N, InpMinSamples);
      return(true);
     }
   double Z[], Y[];
   ArrayResize(Z, N * NF);
   ArrayResize(Y, N);
   double wins = 0;
   for(int k = 0; k < NF; k++)
     {
      double m = 0, v = 0;
      for(int i = 0; i < N; i++)
         m += g_S[i].x[k];
      m /= N;
      for(int i = 0; i < N; i++)
         v += (g_S[i].x[k] - m) * (g_S[i].x[k] - m);
      g_mu[k] = m;
      g_sd[k] = MathSqrt(v / N) + 1e-9;
     }
   for(int i = 0; i < N; i++)
     {
      Y[i] = (g_S[i].pnl > 0) ? 1.0 : 0.0;
      wins += Y[i];
      for(int k = 0; k < NF; k++)
        {
         double z = (g_S[i].x[k] - g_mu[k]) / g_sd[k];
         Z[i * NF + k] = MathMax(-4.0, MathMin(4.0, z));
        }
     }
   g_base = wins / N;
   if(!FitLogistic(Z, Y, N))
     {
      Print("CortexML: logistic fit failed, keeping the previous model");
      return(false);
     }
   g_modelOk = true;
   g_trainN = N;
   g_trainedAt = now;
   //--- in-sample accuracy of the "take" decision (diagnostic only)
   double hit = 0, cnt = 0;
   double xx[NF];
   for(int i = 0; i < N; i++)
     {
      for(int k = 0; k < NF; k++)
         xx[k] = g_S[i].x[k];
      double p = Predict(xx);
      bool take = (g_S[i].dm >= InpCoreDayMove) ? (p > g_base - InpCoreVeto) : (p > g_base + InpMLMargin);
      if(take)
        {
         cnt++;
         hit += Y[i];
        }
     }
   g_trainAcc = (cnt > 0) ? hit / cnt : 0;
   PrintFormat("CortexML: model trained on %d samples (%d fresh), base win %.1f%%, selected win %.1f%% on %d, %u ms",
               N, nb, 100 * g_base, 100 * g_trainAcc, (int)cnt, GetTickCount() - t0);
   SaveMemory();
   return(true);
  }

//+------------------------------------------------------------------+
//| Persistence (live only; the tester always learns from scratch)   |
//+------------------------------------------------------------------+
bool UseFiles() { return(InpPersistMemory && !MQLInfoInteger(MQL_TESTER) && !MQLInfoInteger(MQL_OPTIMIZATION)); }
string MemFile(string kind) { return("CortexML\\" + _Symbol + "_" + IntegerToString(InpMagic) + "_" + kind + ".bin"); }

void SaveMemory()
  {
   if(!UseFiles())
      return;
   int h = FileOpen(MemFile("samples"), FILE_WRITE | FILE_BIN | FILE_COMMON);
   if(h != INVALID_HANDLE)
     {
      FileWriteInteger(h, NF);
      FileWriteArray(h, g_S, 0, g_nS);
      FileClose(h);
     }
   h = FileOpen(MemFile("feedback"), FILE_WRITE | FILE_BIN | FILE_COMMON);
   if(h != INVALID_HANDLE)
     {
      FileWriteArray(h, g_F, 0, g_nF);
      FileClose(h);
     }
  }

void LoadMemory()
  {
   if(!UseFiles())
      return;
   int h = FileOpen(MemFile("samples"), FILE_READ | FILE_BIN | FILE_COMMON);
   if(h != INVALID_HANDLE)
     {
      int nf = FileReadInteger(h);
      if(nf == NF)
        {
         g_nS = (int)FileReadArray(h, g_S);
         if(g_nS < 0)
            g_nS = 0;
        }
      FileClose(h);
     }
   h = FileOpen(MemFile("feedback"), FILE_READ | FILE_BIN | FILE_COMMON);
   if(h != INVALID_HANDLE)
     {
      g_nF = (int)FileReadArray(h, g_F);
      if(g_nF < 0)
         g_nF = 0;
      FileClose(h);
     }
   if(g_nS > 0 || g_nF > 0)
      PrintFormat("CortexML: memory loaded (%d samples, %d real-trade results)", g_nS, g_nF);
  }

//+------------------------------------------------------------------+
//| Trading helpers                                                  |
//+------------------------------------------------------------------+
int CountMyPositions()
  {
   int cnt = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong tk = PositionGetTicket(i);
      if(tk == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == InpMagic)
         cnt++;
     }
   return(cnt);
  }

double NormalizeLot(double lot)
  {
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double mn = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double mx = MathMin(SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX), InpMaxLot);
   if(step <= 0)
      step = 0.01;
   lot = MathFloor(lot / step + 1e-9) * step;
   lot = MathMax(mn, MathMin(mx, lot));
   return(NormalizeDouble(lot, 2));
  }

double CalcLot(int dir)
  {
   double bal = AccountInfoDouble(ACCOUNT_BALANCE);
   if(InpLotMode == LOT_FIXED)
      return(NormalizeLot(InpFixedLot));
   if(InpLotMode == LOT_PER_BALANCE)
      return(NormalizeLot(0.01 * MathFloor(bal / MathMax(InpBalancePer001, 1.0))));
   //--- risk mode: money lost at the hard stop for 1.0 lot
   double price = (dir > 0) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double loss1 = 0;
   ENUM_ORDER_TYPE ot = (dir > 0) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   if(!OrderCalcProfit(ot, _Symbol, 1.0, price, price - dir * InpStopLossUSD, loss1) || loss1 >= 0)
      return(NormalizeLot(InpFixedLot));
   double risk = bal * InpRiskPercent / 100.0;
   return(NormalizeLot(risk / MathAbs(loss1)));
  }

bool DailyLossHit()
  {
   if(InpDailyLossPct <= 0)
      return(false);
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   return(eq < g_dayStartBalance * (1.0 - InpDailyLossPct / 100.0));
  }

bool EquityStopHit()
  {
   if(InpEquityStopPct <= 0)
      return(false);
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   return(eq < g_peakEquity * (1.0 - InpEquityStopPct / 100.0));
  }

void RememberPosition(ulong ticket, double peak, datetime sigT, int sigDir)
  {
   int n = ArraySize(g_posTicket);
   for(int i = 0; i < n; i++)
      if(g_posTicket[i] == ticket)
         return;
   ArrayResize(g_posTicket, n + 1);
   ArrayResize(g_posPeak, n + 1);
   ArrayResize(g_posSigT, n + 1);
   ArrayResize(g_posSigDir, n + 1);
   g_posTicket[n] = ticket;
   g_posPeak[n] = peak;
   g_posSigT[n] = sigT;
   g_posSigDir[n] = sigDir;
  }

int PosIndex(ulong ticket)
  {
   for(int i = ArraySize(g_posTicket) - 1; i >= 0; i--)
      if(g_posTicket[i] == ticket)
         return(i);
   return(-1);
  }

//+------------------------------------------------------------------+
//| Adaptive Profit Shield: runs on every tick                       |
//+------------------------------------------------------------------+
void ManagePositions()
  {
   if(!g_trailOn)
      return;
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double pt  = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double stopLvl = (SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) + 1) * pt;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong tk = PositionGetTicket(i);
      if(tk == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol || PositionGetInteger(POSITION_MAGIC) != InpMagic)
         continue;
      long   type  = PositionGetInteger(POSITION_TYPE);
      double open  = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl    = PositionGetDouble(POSITION_SL);
      double tp    = PositionGetDouble(POSITION_TP);
      double tpDist = (tp > 0) ? MathAbs(tp - open) : InpTakeProfitUSD;
      int k = PosIndex(tk);
      if(k < 0)
        {
         //--- position from a previous session: rebuild the peak from the stop
         double pk = (type == POSITION_TYPE_BUY) ? MathMax(bid, (sl > open) ? sl + g_dist : bid)
                                                  : MathMin(ask, (sl > 0 && sl < open) ? sl - g_dist : ask);
         RememberPosition(tk, pk, 0, 0);
         k = PosIndex(tk);
        }
      if(type == POSITION_TYPE_BUY)
        {
         if(bid > g_posPeak[k])
            g_posPeak[k] = bid;
         double prof = g_posPeak[k] - open;
         if(prof < g_act)
            continue;
         double d = (prof >= InpTightenAt * tpDist) ? g_dist * InpTightenK : g_dist;
         double ns = MathMax(g_posPeak[k] - d, open + g_lock);
         ns = MathMin(ns, bid - stopLvl);
         ns = NormalizeDouble(ns, _Digits);
         if(ns >= sl + InpTrailStepUSD - 1e-9 && ns > open)
            if(!trade.PositionModify(tk, ns, tp) && InpVerbose)
               Print("CortexML: modify failed ", trade.ResultRetcode());
        }
      else
        {
         if(ask < g_posPeak[k])
            g_posPeak[k] = ask;
         double prof = open - g_posPeak[k];
         if(prof < g_act)
            continue;
         double d = (prof >= InpTightenAt * tpDist) ? g_dist * InpTightenK : g_dist;
         double ns = MathMin(g_posPeak[k] + d, open - g_lock);
         ns = MathMax(ns, ask + stopLvl);
         ns = NormalizeDouble(ns, _Digits);
         if((sl <= 0 || ns <= sl - InpTrailStepUSD + 1e-9) && ns < open)
            if(!trade.PositionModify(tk, ns, tp) && InpVerbose)
               Print("CortexML: modify failed ", trade.ResultRetcode());
        }
     }
  }

//+------------------------------------------------------------------+
//| Signal evaluation on every closed M15 bar                        |
//+------------------------------------------------------------------+
void OnNewBar()
  {
   datetime now = TimeCurrent();
   MqlDateTime s;
   TimeToStruct(now, s);
   //--- monthly re-training on the rolling window (past + newest resolved data)
   if(s.year * 12 + s.mon != g_trainMonth)
      Train();
   //--- live window
   if(!LoadSet(now - 40 * 86400, now, g_m15, g_h1, g_h4, g_d1))
      return;
   int i = g_m15.LastClosed(ToNY7(now));
   if(i < 112)
      return;
   double dmRaw = 0;
   if(!DayMove(g_m15, g_d1, i, dmRaw))
      return;
   int dir = (dmRaw > 0) ? 1 : -1;
   double dm = MathAbs(dmRaw);
   g_lastDM = dmRaw;
   g_lastDir = dir;
   double hr = ServerHour(g_m15.t[i]);
   if(hr < InpHourFrom || hr >= InpHourTo)
     {
      g_lastDecision = "outside trading hours";
      return;
     }
   if(dm <= InpMinDayMove)
     {
      g_lastDecision = StringFormat("day move %.2f ATR: no momentum", dmRaw);
      return;
     }
   double x[NF];
   double tpUsed = InpTakeProfitUSD;
   if(!MakeFeatures(g_m15, g_h1, g_h4, g_d1, i, dir, x, tpUsed))
     {
      g_lastDecision = "not enough bars for features";
      return;
     }
   double p = Predict(x);
   g_lastP = p;
   bool take;
   if(g_modelOk)
     {
      g_lastE = p - g_base;
      take = (dm >= InpCoreDayMove) ? (g_lastE > -InpCoreVeto) : (g_lastE > InpMLMargin);
     }
   else
     {
      g_lastE = 0;
      take = (dm >= InpCoreDayMove);
     }
   string tier = (dm >= InpCoreDayMove) ? "core" : "ML tier";
   if(!take)
     {
      g_lastDecision = StringFormat("%s %s skipped (p=%.3f, base=%.3f)", tier, dir > 0 ? "BUY" : "SELL", p, g_base);
      return;
     }
   //--- execution filters
   if(now < InpTradeStart)
     {
      g_lastDecision = "warm-up: trading starts " + TimeToString(InpTradeStart, TIME_DATE);
      return;
     }
   MqlDateTime sb;
   TimeToStruct(g_m15.t[i] + 900, sb);
   if(sb.day_of_week == 5 && sb.hour >= InpFridayLastHour)
      return;
   if(CountMyPositions() >= InpMaxPositions)
      return;
   if(InpMinMinutesGap > 0 && now - g_lastEntry < InpMinMinutesGap * 60)
      return;
   if(DailyLossHit() || EquityStopHit())
     {
      g_lastDecision = "protection: loss limit reached";
      return;
     }
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if(ask - bid > InpMaxSpreadUSD)
     {
      g_lastDecision = StringFormat("spread %.2f too wide", ask - bid);
      return;
     }
   double lot = CalcLot(dir);
   bool ok;
   if(dir > 0)
      ok = trade.Buy(lot, _Symbol, ask, NormalizeDouble(ask - InpStopLossUSD, _Digits),
                     NormalizeDouble(ask + tpUsed, _Digits), InpComment);
   else
      ok = trade.Sell(lot, _Symbol, bid, NormalizeDouble(bid + InpStopLossUSD, _Digits),
                      NormalizeDouble(bid - tpUsed, _Digits), InpComment);
   if(ok && (trade.ResultRetcode() == TRADE_RETCODE_DONE || trade.ResultRetcode() == TRADE_RETCODE_PLACED))
     {
      g_lastEntry = now;
      ulong deal = trade.ResultDeal();
      ulong posId = 0;
      if(deal > 0 && HistoryDealSelect(deal))
         posId = (ulong)HistoryDealGetInteger(deal, DEAL_POSITION_ID);
      if(posId > 0)
         RememberPosition(posId, (dir > 0) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK),
                          g_m15.t[i], dir);
      g_lastDecision = StringFormat("%s %s opened %.2f lot (p=%.3f, base=%.3f, day %.2f ATR)",
                                    tier, dir > 0 ? "BUY" : "SELL", lot, p, g_base, dmRaw);
      if(InpVerbose)
         Print("CortexML: ", g_lastDecision);
     }
   else
      g_lastDecision = "order rejected: " + IntegerToString((int)trade.ResultRetcode());
  }

//+------------------------------------------------------------------+
//| Real trade results -> learning feedback                          |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &request, const MqlTradeResult &result)
  {
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD || !InpLearnFromTrades)
      return;
   if(!HistoryDealSelect(trans.deal))
      return;
   if(HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != InpMagic)
      return;
   if(HistoryDealGetString(trans.deal, DEAL_SYMBOL) != _Symbol)
      return;
   long entry = HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
   if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_OUT_BY)
      return;
   ulong posId = (ulong)HistoryDealGetInteger(trans.deal, DEAL_POSITION_ID);
   int k = PosIndex(posId);
   if(k < 0 || g_posSigT[k] == 0)
      return;
   double vol = HistoryDealGetDouble(trans.deal, DEAL_VOLUME);
   if(vol <= 0)
      return;
   //--- whole position result: profit + swap + commissions of the entry and exit deals
   double money = 0;
   if(HistorySelectByPosition(posId))
     {
      for(int d = HistoryDealsTotal() - 1; d >= 0; d--)
        {
         ulong dt = HistoryDealGetTicket(d);
         if(dt == 0)
            continue;
         money += HistoryDealGetDouble(dt, DEAL_PROFIT) + HistoryDealGetDouble(dt, DEAL_COMMISSION) +
                  HistoryDealGetDouble(dt, DEAL_SWAP);
        }
     }
   else
      money = HistoryDealGetDouble(trans.deal, DEAL_PROFIT) + HistoryDealGetDouble(trans.deal, DEAL_COMMISSION) +
              HistoryDealGetDouble(trans.deal, DEAL_SWAP);
   //--- convert account money into "$ per 0.01 lot" with the symbol's own tick value
   double tv = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double ts = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double perUnit = (ts > 0) ? tv / ts * 0.01 : 1.0;    // money per 1.0 price unit on 0.01 lot
   double pnl001 = money / (vol / 0.01) / MathMax(perUnit, 1e-9);
   ArrayResize(g_F, g_nF + 1);
   g_F[g_nF].t = g_posSigT[k];
   g_F[g_nF].dir = g_posSigDir[k];
   g_F[g_nF].pnl = pnl001;
   g_F[g_nF].te = ToNY7((datetime)HistoryDealGetInteger(trans.deal, DEAL_TIME));
   g_nF++;
   ApplyFeedback();
   if(UseFiles())
      SaveMemory();
  }

//+------------------------------------------------------------------+
//| Panel                                                            |
//+------------------------------------------------------------------+
void DrawPanel()
  {
   if(!InpShowPanel)
      return;
   string mdl = g_modelOk ? StringFormat("online logistic | %d samples | base win %.1f%% | trained %s",
                                         g_trainN, 100 * g_base, TimeToString(g_trainedAt, TIME_DATE))
                          : StringFormat("collecting history (%d / %d samples) - core rule active", g_trainN, InpMinSamples);
   string regime = "flat";
   if(MathAbs(g_lastDM) >= InpCoreDayMove)
      regime = (g_lastDM > 0) ? "strong bullish day" : "strong bearish day";
   else
      if(MathAbs(g_lastDM) >= InpMinDayMove)
         regime = (g_lastDM > 0) ? "bullish drift" : "bearish drift";
   string txt = "CORTEX ML AI  |  Quantix Engineering\n";
   txt += "Model: " + mdl + "\n";
   txt += StringFormat("Day move: %+.2f daily ATR (%s)\n", g_lastDM, regime);
   txt += StringFormat("Last score: p=%.3f  edge=%+.3f\n", g_lastP, g_lastE);
   txt += "Decision: " + g_lastDecision + "\n";
   txt += StringFormat("Exits: SL %.1f | TP %.1f | Shield start %.1f lock %.1f dist %.1f\n",
                       InpStopLossUSD, InpTakeProfitUSD, g_act, g_lock, g_dist);
   txt += StringFormat("Open positions: %d / %d   real results learned: %d", CountMyPositions(), InpMaxPositions, g_nF);
   Comment(txt);
  }

//+------------------------------------------------------------------+
//| Expert lifecycle                                                 |
//+------------------------------------------------------------------+
int OnInit()
  {
   g_shiftZero = (InpServerGMTWinter == 2 && InpServerUsDst);
   switch(InpTrailProfile)
     {
      case TRAIL_SHIELD:
         g_act = 5.0;
         g_lock = 4.0;
         g_dist = 3.0;
         break;
      case TRAIL_BALANCED:
         g_act = 6.0;
         g_lock = 4.0;
         g_dist = 3.0;
         break;
      case TRAIL_RUNNER:
         g_act = 8.0;
         g_lock = 5.0;
         g_dist = 3.0;
         break;
      default:
         g_act = InpTrailStartUSD;
         g_lock = MathMax(3.0, InpTrailLockUSD);
         g_dist = InpTrailDistUSD;
     }
   g_trailOn = (InpTrailProfile != TRAIL_OFF);
   if(g_trailOn && g_lock >= g_act)
     {
      Print("CortexML: trailing lock must be below the activation profit");
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(InpStopLossUSD <= 0 || InpTakeProfitUSD <= 0 || InpTPMinUSD > InpTPMaxUSD)
      return(INIT_PARAMETERS_INCORRECT);
   trade.SetExpertMagicNumber((ulong)InpMagic);
   trade.SetDeviationInPoints(50);
   trade.SetTypeFillingBySymbol(_Symbol);
   g_dayStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   g_peakEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(UseFiles())
      FolderCreate("CortexML", FILE_COMMON);
   LoadMemory();
   Train();
   DrawPanel();
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   SaveMemory();
   Comment("");
  }

void OnTick()
  {
   //--- day bookkeeping for the protection block
   MqlDateTime s;
   TimeToStruct(TimeCurrent(), s);
   int dk = s.year * 400 + s.day_of_year;
   if(dk != g_dayKey)
     {
      g_dayKey = dk;
      g_dayStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
     }
   g_peakEquity = MathMax(g_peakEquity, AccountInfoDouble(ACCOUNT_EQUITY));
   ManagePositions();
   datetime bar = iTime(_Symbol, PERIOD_M15, 0);
   if(bar != 0 && bar != g_lastBar)
     {
      g_lastBar = bar;
      OnNewBar();
      DrawPanel();
     }
  }

double OnTester()
  {
   //--- custom criterion: profit factor weighted by the number of trades (penalises tiny samples)
   double pf = TesterStatistics(STAT_PROFIT_FACTOR);
   double n  = TesterStatistics(STAT_TRADES);
   double dd = TesterStatistics(STAT_EQUITY_DDREL_PERCENT);
   if(n < 30)
      return(0);
   return(pf * MathSqrt(n) / MathMax(1.0, dd));
  }
//+------------------------------------------------------------------+
