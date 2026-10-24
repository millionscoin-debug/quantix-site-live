//+------------------------------------------------------------------+
//|                                                CortexML_AI.mq4   |
//|        Cortex ML AI - XAUUSD intraday-momentum engine with two   |
//|        online-learning models, activity governor, smooth         |
//|        recovery and the Adaptive Profit Shield (MetaTrader 4)    |
//|                                         Quantix Engineering 2026 |
//+------------------------------------------------------------------+
#property copyright   "Quantix Engineering"
#property link        "https://quantixengine.com"
#property version     "1.10"
#property strict
#property description "Cortex ML AI (MT4): the same decision core as the MT5 version (CortexCore.mqh)."
#property description "XAUUSD momentum engine + online logistic models re-trained monthly on the broker's"
#property description "history and the EA's own results. Hard SL $30, TP $12-15, Adaptive Profit Shield,"
#property description "activity governor, smooth post-loss recovery."

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

input string             InpSec1             = "====== Money management ======";   // ----------
input ENUM_LOT_MODE      InpLotMode          = LOT_FIXED;   // Lot mode
input double             InpFixedLot         = 0.01;        // Fixed lot
input double             InpBalancePer001    = 1000.0;      // Balance per 0.01 lot (per-balance mode)
input double             InpRiskPercent      = 1.0;         // Risk % at hard stop (risk mode)
input double             InpMaxLot           = 5.0;         // Maximum lot

input string             InpSec2             = "====== Smooth recovery after losses ======"; // ----------
input bool               InpRecovery         = true;        // Enable smooth recovery
input double             InpRecoveryTrades   = 15.0;        // Spread the deficit over ~N trades
input double             InpRecoveryMaxMult  = 1.5;         // Max lot multiplier while recovering
input double             InpRecoveryUnitUSD  = 6.0;         // Expected win per trade, $ per 0.01 lot

input string             InpSec3             = "====== Exits (USD per 0.01 lot) ======"; // ----------
input double             InpStopLossUSD      = 30.0;        // Hard stop loss
input ENUM_TP_MODE       InpTPMode           = TP_FIXED;    // Take profit mode
input double             InpTakeProfitUSD    = 13.5;        // Fixed take profit
input double             InpTPMinUSD         = 12.0;        // Adaptive TP minimum
input double             InpTPMaxUSD         = 15.0;        // Adaptive TP maximum

input string             InpSec4             = "====== Adaptive Profit Shield ======"; // ----------
input ENUM_TRAIL_PROFILE InpTrailProfile     = TRAIL_SHIELD;// Trailing profile
input double             InpTrailStartUSD    = 5.0;         // Custom: activation profit
input double             InpTrailLockUSD     = 4.0;         // Custom: profit locked at activation (>= 3)
input double             InpTrailDistUSD     = 3.0;         // Custom: distance from the peak
input double             InpTightenAt        = 0.8;         // Tighten when profit >= this share of TP
input double             InpTightenK         = 0.6;         // Distance multiplier near TP
input double             InpTrailStepUSD     = 0.5;         // Minimal SL improvement per modification

input string             InpSec5             = "====== Signal engine ======"; // ----------
input double             InpCoreDayMove      = 1.0;         // Core signal: day move >= x daily ATR
input double             InpMinDayMove       = 0.6;         // ML tier: day move > x daily ATR
input bool               InpUseSessionTier   = true;        // Session tier on quiet days (activity)
input double             InpSessMove6        = 3.0;         // Session trigger: 6h move > x H1 ATR
input double             InpSessMove24       = 7.0;         // Session trigger: 24h move > x H1 ATR
input int                InpHourFrom         = 8;           // First entry hour (NY+7 server clock)
input int                InpHourTo           = 23;          // Entries until hour (exclusive)
input int                InpMaxPositions     = 1;           // Max simultaneous positions
input int                InpMinMinutesGap    = 0;           // Min minutes between entries (0 = off)
input int                InpFridayLastHour   = 23;          // No new entries on Friday from hour

input string             InpSec6             = "====== Machine learning core ======"; // ----------
input bool               InpUseML            = true;        // Use the ML models
input double             InpTrainYears       = 3.0;         // Rolling training window (years)
input int                InpMinSamples       = 300;         // Min labelled samples to activate a model
input double             InpRegC             = 0.05;        // Inverse L2 strength (C)
input double             InpCoreVeto         = 0.03;        // Core veto: skip if p < base - veto
input double             InpMLMargin         = 0.04;        // ML tier: take if p > base + margin
input double             InpSessMargin       = 0.04;        // Session tier: take if p > base + margin
input double             InpIdleDays         = 1.5;         // Activity governor: idle days before relaxing
input double             InpIdleSpanDays     = 2.0;         // ... days to reach the full relaxation
input double             InpIdleRelax        = 0.04;        // ... max margin relaxation (never below base)
input double             InpLabelSpreadUSD   = 0.25;        // Spread assumed when labelling history
input double             InpCommissionPerLot = 7.0;         // Round-turn commission per 1.0 lot
input bool               InpLearnFromTrades  = true;        // Feed real trade results back into the data
input bool               InpPersistMemory    = true;        // Live: keep samples/feedback in Common files

input string             InpSec7             = "====== Protection ======"; // ----------
input double             InpMaxSpreadUSD     = 0.80;        // Max spread at entry
input double             InpDailyLossPct     = 0.0;         // Daily loss limit % (0 = off)
input double             InpEquityStopPct    = 0.0;         // Stop trading below this % drawdown (0 = off)

input string             InpSec8             = "====== Server clock ======"; // ----------
input int                InpServerGMTWinter  = 2;           // Broker GMT offset in winter
input bool               InpServerUsDst      = true;        // Broker follows US daylight saving

input string             InpSec9             = "====== General ======"; // ----------
input int                InpMagic            = 20261024;    // Magic number
input string             InpComment          = "CortexML";  // Order comment
input datetime           InpTradeStart       = D'2000.01.01';// No trading before (tester warm-up)
input int                InpSlippagePoints   = 50;          // Max slippage, points
input bool               InpShowPanel        = true;        // Show info panel
input bool               InpVerbose          = false;       // Verbose log

#include "CortexCore.mqh"

datetime g_lastBar = 0;

//+------------------------------------------------------------------+
//| Platform layer (MT4: orders)                                     |
//+------------------------------------------------------------------+
bool IsMyOrder()
  {
   return(OrderSymbol() == _Symbol && OrderMagicNumber() == InpMagic && OrderType() <= OP_SELL);
  }

int PlatformCountPositions()
  {
   int cnt = 0;
   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
         continue;
      if(IsMyOrder())
         cnt++;
     }
   return(cnt);
  }

void ManagePositions()
  {
   if(!g_trailOn)
      return;
   RefreshRates();
   double bid = MarketInfo(_Symbol, MODE_BID);
   double ask = MarketInfo(_Symbol, MODE_ASK);
   double stopLvl = (MarketInfo(_Symbol, MODE_STOPLEVEL) + 1) * MarketInfo(_Symbol, MODE_POINT);
   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES) || !IsMyOrder())
         continue;
      bool   isBuy = (OrderType() == OP_BUY);
      double open  = OrderOpenPrice();
      double sl    = OrderStopLoss();
      double tp    = OrderTakeProfit();
      int    tk    = OrderTicket();
      int k = PosIndex((ulong)tk);
      if(k < 0)
        {
         //--- order from a previous session: rebuild the peak from the stop
         double pk = isBuy ? MathMax(bid, (sl > open) ? sl + g_dist : bid)
                           : MathMin(ask, (sl > 0 && sl < open) ? sl - g_dist : ask);
         RememberPosition((ulong)tk, pk, 0, 0);
         k = PosIndex((ulong)tk);
        }
      double peak = g_posPeak[k];
      double ns = ShieldNewSL(isBuy, open, sl, tp, bid, ask, peak, stopLvl);
      g_posPeak[k] = peak;
      if(ns > 0)
        {
         if(!OrderModify(tk, open, ns, tp, 0, clrNONE) && InpVerbose)
            Print("CortexML: modify failed ", GetLastError());
        }
     }
  }

//--- detect closed orders of this EA (MT4 has no trade transaction events)
void CheckClosedOrders()
  {
   for(int i = ArraySize(g_posTicket) - 1; i >= 0; i--)
     {
      int tk = (int)g_posTicket[i];
      if(!OrderSelect(tk, SELECT_BY_TICKET))
         continue;
      if(OrderCloseTime() <= 0)
         continue;
      double money = OrderProfit() + OrderCommission() + OrderSwap();
      CoreOnTradeClosed((ulong)tk, money, OrderLots(), OrderCloseTime());
     }
  }

//--- realised P/L history of this EA (for the recovery high-water mark), in closing order
void InitPnlHistory()
  {
   g_eaPnl = 0;
   g_eaHwm = 0;
   int n = OrdersHistoryTotal();
   datetime ct[];
   double   mv[];
   int m = 0;
   ArrayResize(ct, n);
   ArrayResize(mv, n);
   for(int i = 0; i < n; i++)
     {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_HISTORY) || !IsMyOrder())
         continue;
      ct[m] = OrderCloseTime();
      mv[m] = OrderProfit() + OrderCommission() + OrderSwap();
      m++;
     }
   //--- insertion sort by close time
   for(int i = 1; i < m; i++)
     {
      datetime kt = ct[i];
      double km = mv[i];
      int j = i - 1;
      while(j >= 0 && ct[j] > kt)
        {
         ct[j + 1] = ct[j];
         mv[j + 1] = mv[j];
         j--;
        }
      ct[j + 1] = kt;
      mv[j + 1] = km;
     }
   for(int i = 0; i < m; i++)
     {
      g_eaPnl += mv[i];
      g_eaHwm = MathMax(g_eaHwm, g_eaPnl);
     }
  }

int SendMarket(int dir, double lot, double tpDist)
  {
   RefreshRates();
   double price = (dir > 0) ? MarketInfo(_Symbol, MODE_ASK) : MarketInfo(_Symbol, MODE_BID);
   double sl = NormalizeDouble(price - dir * InpStopLossUSD, _Digits);
   double tp = NormalizeDouble(price + dir * tpDist, _Digits);
   int type = (dir > 0) ? OP_BUY : OP_SELL;
   color clr = (dir > 0) ? clrDodgerBlue : clrOrangeRed;
   int ticket = OrderSend(_Symbol, type, lot, price, InpSlippagePoints, sl, tp, InpComment, InpMagic, 0, clr);
   if(ticket < 0)
     {
      int err = GetLastError();
      //--- market-execution brokers may reject stops in the opening request: open naked, then attach
      if(err == ERR_INVALID_STOPS || err == ERR_INVALID_TRADE_PARAMETERS)
        {
         RefreshRates();
         price = (dir > 0) ? MarketInfo(_Symbol, MODE_ASK) : MarketInfo(_Symbol, MODE_BID);
         ticket = OrderSend(_Symbol, type, lot, price, InpSlippagePoints, 0, 0, InpComment, InpMagic, 0, clr);
         if(ticket > 0 && OrderSelect(ticket, SELECT_BY_TICKET))
           {
            double op = OrderOpenPrice();
            if(!OrderModify(ticket, op, NormalizeDouble(op - dir * InpStopLossUSD, _Digits),
                            NormalizeDouble(op + dir * tpDist, _Digits), 0, clrNONE))
               Print("CortexML: could not attach SL/TP to ticket ", ticket, " error ", GetLastError());
           }
        }
      else
         g_lastDecision = "order rejected: " + IntegerToString(err);
     }
   return(ticket);
  }

void OnNewBar()
  {
   CortexSignal sig;
   if(!CoreEvaluate(sig))
      return;
   if(!CoreTimeFilters(sig))
      return;
   if(PlatformCountPositions() >= InpMaxPositions)
      return;
   if(CoreProtectionHit())
      return;
   double bid = MarketInfo(_Symbol, MODE_BID);
   double ask = MarketInfo(_Symbol, MODE_ASK);
   if(ask - bid > InpMaxSpreadUSD)
     {
      g_lastDecision = StringFormat("spread %.2f too wide", ask - bid);
      return;
     }
   double base = BaseLot();
   double mult = RecoveryMultiplier(base);
   double lot = NormalizeLot(base * mult);
   int ticket = SendMarket(sig.dir, lot, sig.tp);
   if(ticket > 0)
     {
      RefreshRates();
      RememberPosition((ulong)ticket, sig.dir > 0 ? MarketInfo(_Symbol, MODE_BID) : MarketInfo(_Symbol, MODE_ASK),
                       sig.sigT, sig.dir);
      CoreOnEntry(sig, lot, mult);
     }
  }

//+------------------------------------------------------------------+
//| Expert lifecycle                                                 |
//+------------------------------------------------------------------+
int OnInit()
  {
   g_shiftZero = (InpServerGMTWinter == 2 && InpServerUsDst);
   if(!CoreSetupShield())
      return(INIT_PARAMETERS_INCORRECT);
   g_dayStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   g_peakEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   g_lastEntry = TimeCurrent();
   LoadMemory();
   InitPnlHistory();
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
   CoreBookkeeping();
   CheckClosedOrders();
   ManagePositions();
   datetime bar = iTime(_Symbol, PERIOD_M15, 0);
   if(bar != 0 && bar != g_lastBar)
     {
      g_lastBar = bar;
      OnNewBar();
      DrawPanel();
     }
  }
//+------------------------------------------------------------------+
