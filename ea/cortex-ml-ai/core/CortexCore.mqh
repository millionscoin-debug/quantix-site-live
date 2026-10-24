//+------------------------------------------------------------------+
//|                                                 CortexCore.mqh   |
//|  Platform-neutral core of Cortex ML AI (shared by MT5 and MT4).  |
//|  The including expert must declare the inputs listed in the      |
//|  README (same names in both platforms) before including this     |
//|  file, and implement PlatformCountPositions().                   |
//|                                         Quantix Engineering 2026 |
//+------------------------------------------------------------------+
#define CORTEX_VERSION "1.10"
//--- number of model features (must match research/features.py)
#define NF 67
//--- model slots: 0 = day-momentum tier (core + ML tier), 1 = session tier
#define NMODELS 2

//--- implemented by the platform wrapper
int  PlatformCountPositions();
//--- forward declarations inside the core
void SaveMemory();
bool Train();

//+------------------------------------------------------------------+
//| Trade-management constants derived from the inputs               |
//+------------------------------------------------------------------+
double g_act = 5.0, g_lock = 4.0, g_dist = 3.0;
bool   g_trailOn = true;

bool CoreSetupShield()
  {
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
      return(false);
     }
   return(InpStopLossUSD > 0 && InpTakeProfitUSD > 0 && InpTPMinUSD <= InpTPMaxUSD);
  }

//+------------------------------------------------------------------+
//| Time helpers: features are computed on the "NY+7" clock          |
//| (UTC+2 winter / UTC+3 US summer), the clock of IC Markets & co.  |
//+------------------------------------------------------------------+
int      g_dstYear = -1;
datetime g_dstFrom = 0, g_dstTo = 0;
bool     g_shiftZero = true;

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

double ClockHour(datetime t)
  {
   MqlDateTime s;
   TimeToStruct(t, s);
   return(s.hour + s.min / 60.0);
  }

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
   double            dayOpen[], dayHigh[], dayLow[];

                     CTF() { n = 0; dur = 60; }

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
   datetime m1Start = (ok1 && m1.n > 0) ? m1.t[0] : ToNY7(to);
   t_path.Reset(60);
   if(m1Start > ToNY7(from) + 86400)
     {
      if(LoadTF(PERIOD_M5, from, to, m5, 300))
         for(int i = 0; i < m5.n && m5.t[i] < m1Start; i++)
            t_path.Add(m5.t[i], m5.o[i], m5.h[i], m5.l[i], m5.c[i]);
     }
   if(ok1)
      for(int i = 0; i < m1.n; i++)
         t_path.Add(m1.t[i], m1.o[i], m1.h[i], m1.l[i], m1.c[i]);
   //--- never label on a bar that is still forming (MT4 tester may expose the whole minute)
   datetime nowNY = ToNY7(TimeCurrent());
   while(t_path.n > 0 && t_path.t[t_path.n - 1] + 60 > nowNY)
      t_path.n--;
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

//--- raw 6h / 24h H1 moves in H1 ATR (session tier trigger)
bool SessionMoves(CTF &m15, CTF &h1, int i, double &r6, double &r24)
  {
   int jh = h1.LastClosed(m15.t[i] + 900);
   double v[19];
   if(jh < 112 || !h1.Block(jh, v))
      return(false);
   r6 = v[2];
   r24 = v[4];
   return(true);
  }

bool SessionTrigger(double r6, double r24, int dir)
  {
   return(dir * r6 > InpSessMove6 || dir * r24 > InpSessMove24);
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
   x[p++] = ClockHour(m15.t[i]);
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

void SimTrade(CTF &path, int i0, int dir, double tpDist, SimResult &r)
  {
   double spr  = InpLabelSpreadUSD;
   double comm = InpCommissionPerLot / 100.0;      // per 0.01 lot
   double q = (dir > 0) ? 0.0 : spr;               // exit-side quote offset
   double entry = path.o[i0] + ((dir > 0) ? spr : 0.0);
   double slv = entry - dir * InpStopLossUSD;
   double hard = slv;
   double tpv = entry + dir * tpDist;
   double peak = entry;
   double pts[4];
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
            return;
           }
         if((dir > 0 && oo >= tpv) || (dir < 0 && oo <= tpv))
           {
            r.pnl = dir * (oo - entry) - comm;
            r.exitIdx = i;
            r.reason = 1;
            return;
           }
        }
      pts[0] = oo;
      pts[1] = (cc >= oo) ? ll : hh;
      pts[2] = (cc >= oo) ? hh : ll;
      pts[3] = cc;
      for(int k = 0; k < 3; k++)
        {
         double px = 0;
         if(Leg(dir, pts[k], pts[k + 1], entry, slv, tpv, peak, tpDist, px))
           {
            r.pnl = dir * (px - entry) - comm;
            r.exitIdx = i;
            r.reason = (px == tpv) ? 1 : ((slv == hard) ? 2 : 3);
            return;
           }
        }
     }
   r.pnl = dir * (path.c[path.n - 1] + q - entry) - comm;
   r.exitIdx = path.n - 1;
   r.reason = 5;
  }

//--- Adaptive Profit Shield for a live position: returns the new stop or 0 (no change)
double ShieldNewSL(bool isBuy, double open, double sl, double tp, double bid, double ask, double &peak, double stopLvl)
  {
   if(!g_trailOn)
      return(0);
   double tpDist = (tp > 0) ? MathAbs(tp - open) : InpTakeProfitUSD;
   if(isBuy)
     {
      if(bid > peak)
         peak = bid;
      double prof = peak - open;
      if(prof < g_act)
         return(0);
      double d = (prof >= InpTightenAt * tpDist) ? g_dist * InpTightenK : g_dist;
      double ns = MathMax(peak - d, open + g_lock);
      ns = MathMin(ns, bid - stopLvl);
      ns = NormalizeDouble(ns, _Digits);
      if(ns >= sl + InpTrailStepUSD - 1e-9 && ns > open)
         return(ns);
      return(0);
     }
   if(ask < peak)
      peak = ask;
   double prof = open - peak;
   if(prof < g_act)
      return(0);
   double d = (prof >= InpTightenAt * tpDist) ? g_dist * InpTightenK : g_dist;
   double ns = MathMin(peak + d, open - g_lock);
   ns = MathMax(ns, ask + stopLvl);
   ns = NormalizeDouble(ns, _Digits);
   if((sl <= 0 || ns <= sl - InpTrailStepUSD + 1e-9) && ns < open)
      return(ns);
   return(0);
  }

//+------------------------------------------------------------------+
//| Labelled sample memory                                           |
//+------------------------------------------------------------------+
struct Sample
  {
   datetime          t;        // signal bar start (NY+7 clock)
   datetime          te;       // exit time of the (simulated or real) trade
   int               dir;
   int               tier;     // 0 day-momentum, 1 session
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

//--- models
double   g_mu[NMODELS][NF], g_sd[NMODELS][NF], g_w[NMODELS][NF];
double   g_b[NMODELS], g_base[NMODELS];
bool     g_modelOk[NMODELS];
int      g_trainN[NMODELS];
datetime g_trainedAt = 0;
int      g_trainMonth = -1;

//--- runtime state shared with the wrappers
datetime g_lastEntry = 0;
double   g_lastP = -1, g_lastE = 0, g_lastDM = 0, g_lastMult = 1.0;
string   g_lastDecision = "waiting for the first closed bar";
double   g_eaPnl = 0, g_eaHwm = 0;     // realised P/L of this EA (money) and its high-water mark

//--- positions opened by this EA (ticket / position id -> signal) for trailing peaks and feedback
ulong    g_posTicket[];
double   g_posPeak[];
datetime g_posSigT[];
int      g_posSigDir[];

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

void ForgetPosition(ulong ticket)
  {
   int k = PosIndex(ticket);
   int n = ArraySize(g_posTicket);
   if(k < 0)
      return;
   g_posTicket[k] = g_posTicket[n - 1];
   g_posPeak[k] = g_posPeak[n - 1];
   g_posSigT[k] = g_posSigT[n - 1];
   g_posSigDir[k] = g_posSigDir[n - 1];
   ArrayResize(g_posTicket, n - 1);
   ArrayResize(g_posPeak, n - 1);
   ArrayResize(g_posSigT, n - 1);
   ArrayResize(g_posSigDir, n - 1);
  }

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

bool FitLogistic(int m, const double &Z[], const double &Y[], int N)
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
            for(int c = 0; c <= k; c++)
               H[row + c] += wz * Z[off + c];
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
         for(int c = 0; c < k; c++)
            H[c * d + k] = H[k * d + c];
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
      g_w[m][k] = W[k];
   g_b[m] = W[NF];
   return(true);
  }

double Predict(int m, const double &x[])
  {
   if(!g_modelOk[m])
      return(-1);
   double eta = g_b[m];
   for(int k = 0; k < NF; k++)
     {
      double z = (x[k] - g_mu[m][k]) / g_sd[m][k];
      z = MathMax(-4.0, MathMin(4.0, z));
      eta += g_w[m][k] * z;
     }
   return(Sigmoid(eta));
  }

//+------------------------------------------------------------------+
//| Build the labelled dataset from history and train both models    |
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

bool InTradingHours(datetime barStart)
  {
   double hr = ClockHour(barStart);
   return(hr >= InpHourFrom && hr < InpHourTo);
  }

void AddBuilt(Sample &arr[], int &nb, datetime t, datetime te, int dir, int tier, double dm, double pnl, const double &x[])
  {
   ArrayResize(arr, nb + 1, 20000);
   arr[nb].t = t;
   arr[nb].te = te;
   arr[nb].dir = dir;
   arr[nb].tier = tier;
   arr[nb].dm = dm;
   arr[nb].pnl = pnl;
   arr[nb].real = 0;
   for(int k = 0; k < NF; k++)
      arr[nb].x[k] = x[k];
   nb++;
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
   SimResult r;
   for(int i = 112; i < t_m15.n - 1; i++)
     {
      if(!InTradingHours(t_m15.t[i]))
         continue;
      double dmRaw = 0;
      if(!DayMove(t_m15, t_d1, i, dmRaw))
         continue;
      datetime e = t_m15.t[i] + 900;
      int i0 = t_path.FirstAtOrAfter(e);
      if(i0 < 0 || t_path.t[i0] - e >= 3600)
         continue;
      int dirs[2];
      int nd = 0, tier = 0;
      if(MathAbs(dmRaw) > InpMinDayMove)
         dirs[nd++] = (dmRaw > 0) ? 1 : -1;
      else
        {
         if(!InpUseSessionTier)
            continue;
         double r6 = 0, r24 = 0;
         if(!SessionMoves(t_m15, t_h1, i, r6, r24))
            continue;
         tier = 1;
         if(SessionTrigger(r6, r24, -1))
            dirs[nd++] = -1;
         if(SessionTrigger(r6, r24, 1))
            dirs[nd++] = 1;
        }
      for(int q = 0; q < nd; q++)
        {
         int dir = dirs[q];
         double tpUsed = InpTakeProfitUSD;
         if(!MakeFeatures(t_m15, t_h1, t_h4, t_d1, i, dir, x, tpUsed))
            continue;
         SimTrade(t_path, i0, dir, tpUsed, r);
         if(r.reason == 5)
            continue;                          // still open at the end of the data
         datetime te = t_path.t[r.exitIdx];
         if(te >= nowNY)
            continue;
         AddBuilt(built, nb, t_m15.t[i], te, dir, tier, dir * dmRaw, r.pnl, x);
        }
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

bool TrainModel(int m)
  {
   int N = 0;
   for(int i = 0; i < g_nS; i++)
      if(g_S[i].tier == m)
         N++;
   g_trainN[m] = N;
   if(!InpUseML || N < InpMinSamples)
     {
      g_modelOk[m] = false;
      return(false);
     }
   double Z[], Y[];
   ArrayResize(Z, N * NF);
   ArrayResize(Y, N);
   int idx[];
   ArrayResize(idx, N);
   int q = 0;
   for(int i = 0; i < g_nS; i++)
      if(g_S[i].tier == m)
         idx[q++] = i;
   for(int k = 0; k < NF; k++)
     {
      double mean = 0, var = 0;
      for(int j = 0; j < N; j++)
         mean += g_S[idx[j]].x[k];
      mean /= N;
      for(int j = 0; j < N; j++)
         var += (g_S[idx[j]].x[k] - mean) * (g_S[idx[j]].x[k] - mean);
      g_mu[m][k] = mean;
      g_sd[m][k] = MathSqrt(var / N) + 1e-9;
     }
   double wins = 0;
   for(int j = 0; j < N; j++)
     {
      int i = idx[j];
      Y[j] = (g_S[i].pnl > 0) ? 1.0 : 0.0;
      wins += Y[j];
      for(int k = 0; k < NF; k++)
        {
         double z = (g_S[i].x[k] - g_mu[m][k]) / g_sd[m][k];
         Z[j * NF + k] = MathMax(-4.0, MathMin(4.0, z));
        }
     }
   g_base[m] = wins / N;
   if(!FitLogistic(m, Z, Y, N))
     {
      PrintFormat("CortexML: logistic fit failed for model %d, keeping the previous one", m);
      return(false);
     }
   g_modelOk[m] = true;
   return(true);
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
   MqlDateTime s;
   TimeToStruct(now, s);
   g_trainMonth = s.year * 12 + s.mon;
   g_trainedAt = now;
   for(int m = 0; m < NMODELS; m++)
      TrainModel(m);
   PrintFormat("CortexML %s: trained on %d samples (%d fresh). day-tier model %s n=%d base %.1f%% | session model %s n=%d base %.1f%% | %u ms",
               CORTEX_VERSION, g_nS, nb, g_modelOk[0] ? "ON" : "off", g_trainN[0], 100 * g_base[0],
               g_modelOk[1] ? "ON" : "off", g_trainN[1], 100 * g_base[1], GetTickCount() - t0);
   SaveMemory();
   return(true);
  }

//+------------------------------------------------------------------+
//| Persistence (live only; the tester always learns from scratch)   |
//+------------------------------------------------------------------+
bool UseFiles() { return(InpPersistMemory && !MQLInfoInteger(MQL_TESTER) && !MQLInfoInteger(MQL_OPTIMIZATION)); }
string MemFile(string kind) { return("CortexML\\" + _Symbol + "_" + IntegerToString(InpMagic) + "_" + kind + "_v2.bin"); }

void SaveMemory()
  {
   if(!UseFiles())
      return;
   int h = FileOpen(MemFile("samples"), FILE_WRITE | FILE_BIN | FILE_COMMON);
   if(h != INVALID_HANDLE)
     {
      FileWriteInteger(h, NF);
      if(g_nS > 0)
         FileWriteArray(h, g_S, 0, g_nS);
      FileClose(h);
     }
   h = FileOpen(MemFile("feedback"), FILE_WRITE | FILE_BIN | FILE_COMMON);
   if(h != INVALID_HANDLE)
     {
      if(g_nF > 0)
         FileWriteArray(h, g_F, 0, g_nF);
      FileClose(h);
     }
  }

void LoadMemory()
  {
   if(!UseFiles())
      return;
   FolderCreate("CortexML", FILE_COMMON);
   int h = FileOpen(MemFile("samples"), FILE_READ | FILE_BIN | FILE_COMMON);
   if(h != INVALID_HANDLE)
     {
      int nf = FileReadInteger(h);
      if(nf == NF && !FileIsEnding(h))
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
      if(!FileIsEnding(h))
        {
         g_nF = (int)FileReadArray(h, g_F);
         if(g_nF < 0)
            g_nF = 0;
        }
      FileClose(h);
     }
   if(g_nS > 0 || g_nF > 0)
      PrintFormat("CortexML: memory loaded (%d samples, %d real-trade results)", g_nS, g_nF);
  }

//+------------------------------------------------------------------+
//| Real trade results -> learning feedback and recovery state       |
//+------------------------------------------------------------------+
//--- money per 1.0 of price distance on 0.01 lot (XAUUSD: $1)
double MoneyPerUnit001()
  {
   double tv = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double ts = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   return((ts > 0 && tv > 0) ? tv / ts * 0.01 : 1.0);
  }

void CoreOnTradeClosed(ulong ticket, double money, double volume, datetime closeTimeServer)
  {
   g_eaPnl += money;
   g_eaHwm = MathMax(g_eaHwm, g_eaPnl);
   int k = PosIndex(ticket);
   if(k >= 0 && InpLearnFromTrades && g_posSigT[k] > 0 && volume > 0)
     {
      double pnl001 = money / (volume / 0.01) / MathMax(MoneyPerUnit001(), 1e-9);
      ArrayResize(g_F, g_nF + 1);
      g_F[g_nF].t = g_posSigT[k];
      g_F[g_nF].dir = g_posSigDir[k];
      g_F[g_nF].pnl = pnl001;
      g_F[g_nF].te = ToNY7(closeTimeServer);
      g_nF++;
      ApplyFeedback();
      if(UseFiles())
         SaveMemory();
     }
   if(k >= 0)
      ForgetPosition(ticket);
  }

//+------------------------------------------------------------------+
//| Money management + smooth recovery                               |
//+------------------------------------------------------------------+
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

double BaseLot()
  {
   double bal = AccountInfoDouble(ACCOUNT_BALANCE);
   if(InpLotMode == LOT_FIXED)
      return(InpFixedLot);
   if(InpLotMode == LOT_PER_BALANCE)
      return(MathMax(0.01, 0.01 * MathFloor(bal / MathMax(InpBalancePer001, 1.0))));
   //--- risk mode: money lost at the hard stop for 1.0 lot
   double ts = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tv = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   if(ts <= 0 || tv <= 0)
      return(InpFixedLot);
   double loss1 = InpStopLossUSD / ts * tv;
   return(bal * InpRiskPercent / 100.0 / loss1);
  }

//--- deficit of this EA under its high-water mark, spread over K future trades (capped multiplier)
double RecoveryMultiplier(double baseLot)
  {
   if(!InpRecovery || baseLot <= 0)
      return(1.0);
   double deficit = g_eaHwm - g_eaPnl;
   if(deficit <= 0)
      return(1.0);
   double deficit001 = deficit / ((baseLot / 0.01) * MathMax(MoneyPerUnit001(), 1e-9));
   double mult = 1.0 + deficit001 / (MathMax(InpRecoveryTrades, 1.0) * MathMax(InpRecoveryUnitUSD, 0.5));
   mult = MathMin(mult, MathMax(1.0, InpRecoveryMaxMult));
   return(MathRound(mult * 100.0) / 100.0);
  }

//+------------------------------------------------------------------+
//| Signal evaluation on every closed M15 bar                        |
//+------------------------------------------------------------------+
struct CortexSignal
  {
   int               dir;
   int               tier;     // 0 core, 1 ML tier, 2 session tier
   double            p;
   double            base;
   double            dm;
   double            tp;
   datetime          sigT;     // signal bar start (NY+7 clock)
  };

string TierName(int tier)
  {
   if(tier == 0)
      return("core");
   if(tier == 1)
      return("ML tier");
   return("session tier");
  }

//--- activity governor: after InpIdleDays without entries the ML margins relax linearly
double IdleRelax()
  {
   if(InpIdleDays <= 0)
      return(0);
   datetime ref = (g_lastEntry > InpTradeStart) ? g_lastEntry : InpTradeStart;   // warm-up does not count as idle
   double idle = (double)(TimeCurrent() - ref) / 86400.0;
   double r = (idle - InpIdleDays) / MathMax(InpIdleSpanDays, 0.1);
   return(MathMax(0.0, MathMin(1.0, r)) * InpIdleRelax);
  }

//--- returns true when the model wants a trade (platform filters are applied by the wrapper)
bool CoreEvaluate(CortexSignal &sig)
  {
   datetime now = TimeCurrent();
   MqlDateTime s;
   TimeToStruct(now, s);
   //--- monthly re-training on the rolling window (past + newest resolved data + real results)
   if(s.year * 12 + s.mon != g_trainMonth)
      Train();
   if(!LoadSet(now - 40 * 86400, now, g_m15, g_h1, g_h4, g_d1))
      return(false);
   int i = g_m15.LastClosed(ToNY7(now));
   if(i < 112)
      return(false);
   double dmRaw = 0;
   if(!DayMove(g_m15, g_d1, i, dmRaw))
      return(false);
   g_lastDM = dmRaw;
   if(!InTradingHours(g_m15.t[i]))
     {
      g_lastDecision = "outside trading hours";
      return(false);
     }
   double relax = IdleRelax();
   double x[NF];
   double tpUsed = InpTakeProfitUSD;
   sig.sigT = g_m15.t[i];
   //--- tier 0/1: strong or moderate day move in the day's direction
   if(MathAbs(dmRaw) > InpMinDayMove)
     {
      int dir = (dmRaw > 0) ? 1 : -1;
      double dm = MathAbs(dmRaw);
      if(!MakeFeatures(g_m15, g_h1, g_h4, g_d1, i, dir, x, tpUsed))
        {
         g_lastDecision = "not enough bars for features";
         return(false);
        }
      double p = Predict(0, x);
      bool take;
      int tier = (dm >= InpCoreDayMove) ? 0 : 1;
      if(g_modelOk[0])
        {
         double e = p - g_base[0];
         g_lastE = e;
         if(tier == 0)
            take = (e > -InpCoreVeto);
         else
            take = (e > MathMax(InpMLMargin - relax, 0.0));
        }
      else
        {
         g_lastE = 0;
         take = (tier == 0);
        }
      g_lastP = p;
      if(!take)
        {
         g_lastDecision = StringFormat("%s %s skipped (p=%.3f, base=%.3f)", TierName(tier), dir > 0 ? "BUY" : "SELL", p, g_base[0]);
         return(false);
        }
      sig.dir = dir;
      sig.tier = tier;
      sig.p = p;
      sig.base = g_base[0];
      sig.dm = dmRaw;
      sig.tp = tpUsed;
      return(true);
     }
   //--- tier 2: quiet day, strong 6h / 24h session move, separate model
   if(!InpUseSessionTier || !g_modelOk[1])
     {
      g_lastDecision = StringFormat("day move %.2f ATR: no momentum", dmRaw);
      return(false);
     }
   double r6 = 0, r24 = 0;
   if(!SessionMoves(g_m15, g_h1, i, r6, r24))
      return(false);
   double bestE = -1, bestP = -1, bestTp = InpTakeProfitUSD;
   int bestDir = 0;
   for(int q = -1; q <= 1; q += 2)
     {
      if(!SessionTrigger(r6, r24, q))
         continue;
      if(!MakeFeatures(g_m15, g_h1, g_h4, g_d1, i, q, x, tpUsed))
         continue;
      double p = Predict(1, x);
      double e = p - g_base[1];
      if(bestDir == 0 || e > bestE)
        {
         bestE = e;
         bestP = p;
         bestDir = q;
         bestTp = tpUsed;
        }
     }
   if(bestDir == 0)
     {
      g_lastDecision = StringFormat("day move %.2f ATR, no session impulse", dmRaw);
      return(false);
     }
   g_lastP = bestP;
   g_lastE = bestE;
   if(bestE <= MathMax(InpSessMargin - relax, 0.0))
     {
      g_lastDecision = StringFormat("session tier %s skipped (p=%.3f, base=%.3f)", bestDir > 0 ? "BUY" : "SELL", bestP, g_base[1]);
      return(false);
     }
   sig.dir = bestDir;
   sig.tier = 2;
   sig.p = bestP;
   sig.base = g_base[1];
   sig.dm = dmRaw;
   sig.tp = bestTp;
   return(true);
  }

//--- common execution filters (time based); position / spread checks are done by the wrapper
bool CoreTimeFilters(CortexSignal &sig)
  {
   datetime now = TimeCurrent();
   if(now < InpTradeStart)
     {
      g_lastDecision = "warm-up: trading starts " + TimeToString(InpTradeStart, TIME_DATE);
      return(false);
     }
   MqlDateTime sb;
   TimeToStruct(sig.sigT + 900, sb);
   if(sb.day_of_week == 5 && sb.hour >= InpFridayLastHour)
     {
      g_lastDecision = "Friday cut-off";
      return(false);
     }
   if(InpMinMinutesGap > 0 && now - g_lastEntry < InpMinMinutesGap * 60)
      return(false);
   return(true);
  }

void CoreOnEntry(CortexSignal &sig, double lot, double mult)
  {
   g_lastEntry = TimeCurrent();
   g_lastMult = mult;
   g_lastDecision = StringFormat("%s %s opened %.2f lot (x%.2f) p=%.3f base=%.3f day %.2f ATR",
                                 TierName(sig.tier), sig.dir > 0 ? "BUY" : "SELL", lot, mult, sig.p, sig.base, sig.dm);
   if(InpVerbose)
      Print("CortexML: ", g_lastDecision);
  }

//+------------------------------------------------------------------+
//| Account protection                                               |
//+------------------------------------------------------------------+
double g_dayStartBalance = 0;
int    g_dayKey = -1;
double g_peakEquity = 0;

void CoreBookkeeping()
  {
   MqlDateTime s;
   TimeToStruct(TimeCurrent(), s);
   int dk = s.year * 400 + s.day_of_year;
   if(dk != g_dayKey)
     {
      g_dayKey = dk;
      g_dayStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
     }
   g_peakEquity = MathMax(g_peakEquity, AccountInfoDouble(ACCOUNT_EQUITY));
  }

bool CoreProtectionHit()
  {
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(InpDailyLossPct > 0 && eq < g_dayStartBalance * (1.0 - InpDailyLossPct / 100.0))
     {
      g_lastDecision = "protection: daily loss limit";
      return(true);
     }
   if(InpEquityStopPct > 0 && eq < g_peakEquity * (1.0 - InpEquityStopPct / 100.0))
     {
      g_lastDecision = "protection: equity stop";
      return(true);
     }
   return(false);
  }

//+------------------------------------------------------------------+
//| Panel                                                            |
//+------------------------------------------------------------------+
void DrawPanel()
  {
   if(!InpShowPanel)
      return;
   string m0 = g_modelOk[0] ? StringFormat("ON n=%d base %.1f%%", g_trainN[0], 100 * g_base[0])
                            : StringFormat("learning %d/%d (core rule active)", g_trainN[0], InpMinSamples);
   string m1 = g_modelOk[1] ? StringFormat("ON n=%d base %.1f%%", g_trainN[1], 100 * g_base[1])
                            : StringFormat("learning %d/%d", g_trainN[1], InpMinSamples);
   string regime = "quiet day";
   if(MathAbs(g_lastDM) >= InpCoreDayMove)
      regime = (g_lastDM > 0) ? "strong bullish day" : "strong bearish day";
   else
      if(MathAbs(g_lastDM) > InpMinDayMove)
         regime = (g_lastDM > 0) ? "bullish drift" : "bearish drift";
   double deficit = g_eaHwm - g_eaPnl;
   string txt = "CORTEX ML AI " + CORTEX_VERSION + "  |  Quantix Engineering\n";
   txt += "Day model: " + m0 + "   Session model: " + m1 + "\n";
   txt += "Last training: " + TimeToString(g_trainedAt, TIME_DATE) + StringFormat("   idle relax: %.3f\n", IdleRelax());
   txt += StringFormat("Day move: %+.2f daily ATR (%s)\n", g_lastDM, regime);
   txt += StringFormat("Last score: p=%.3f  edge=%+.3f\n", g_lastP, g_lastE);
   txt += "Decision: " + g_lastDecision + "\n";
   txt += StringFormat("Shield: SL %.1f | TP %.1f | start %.1f lock %.1f dist %.1f\n", InpStopLossUSD, InpTakeProfitUSD, g_act, g_lock, g_dist);
   txt += StringFormat("Recovery: %s  deficit %.2f  next lot x%.2f\n", InpRecovery ? "ON" : "off", MathMax(deficit, 0.0), RecoveryMultiplier(BaseLot()));
   txt += StringFormat("Open positions: %d / %d   real results learned: %d", PlatformCountPositions(), InpMaxPositions, g_nF);
   Comment(txt);
  }

//+------------------------------------------------------------------+
