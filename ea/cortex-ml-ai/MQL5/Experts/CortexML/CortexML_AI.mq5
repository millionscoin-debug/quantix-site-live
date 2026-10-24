//+------------------------------------------------------------------+
//|                                                CortexML_AI.mq5   |
//|        Cortex ML AI - XAUUSD intraday-momentum engine with two   |
//|        online-learning models, activity governor, smooth         |
//|        recovery and the Adaptive Profit Shield (MetaTrader 5)    |
//|                                         Quantix Engineering 2026 |
//+------------------------------------------------------------------+
#property copyright   "Quantix Engineering"
#property link        "https://quantixengine.com"
#property version     "1.10"
#property description "Cortex ML AI: XAUUSD momentum engine + online logistic models re-trained monthly"
#property description "on the broker's own history and on the EA's real trade results."
#property description "Hard SL $30, TP $12-15, Adaptive Profit Shield (locks >= $3), activity governor,"
#property description "smooth post-loss recovery. Logic shared 1:1 with the MT4 version (CortexCore.mqh)."

#include <Trade\Trade.mqh>

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

input group "=== Money management ==="
input ENUM_LOT_MODE      InpLotMode          = LOT_FIXED;   // Lot mode
input double             InpFixedLot         = 0.01;        // Fixed lot
input double             InpBalancePer001    = 1000.0;      // Balance per 0.01 lot (per-balance mode)
input double             InpRiskPercent      = 1.0;         // Risk % at hard stop (risk mode)
input double             InpMaxLot           = 5.0;         // Maximum lot

input group "=== Smooth recovery after losses ==="
input bool               InpRecovery         = true;        // Enable smooth recovery
input double             InpRecoveryTrades   = 15.0;        // Spread the deficit over ~N trades
input double             InpRecoveryMaxMult  = 1.5;         // Max lot multiplier while recovering
input double             InpRecoveryUnitUSD  = 6.0;         // Expected win per trade, $ per 0.01 lot

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

input group "=== Signal engine ==="
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

input group "=== Machine learning core ==="
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

#include "CortexCore.mqh"

CTrade   trade;
datetime g_lastBar = 0;

//+------------------------------------------------------------------+
//| Platform layer (MT5: positions / deals)                          |
//+------------------------------------------------------------------+
int PlatformCountPositions()
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

void ManagePositions()
  {
   if(!g_trailOn)
      return;
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double stopLvl = (SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) + 1) * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong tk = PositionGetTicket(i);
      if(tk == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol || PositionGetInteger(POSITION_MAGIC) != InpMagic)
         continue;
      bool   isBuy = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY);
      double open  = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl    = PositionGetDouble(POSITION_SL);
      double tp    = PositionGetDouble(POSITION_TP);
      ulong  id    = (ulong)PositionGetInteger(POSITION_IDENTIFIER);
      int k = PosIndex(id);
      if(k < 0)
        {
         //--- position from a previous session: rebuild the peak from the stop
         double pk = isBuy ? MathMax(bid, (sl > open) ? sl + g_dist : bid)
                           : MathMin(ask, (sl > 0 && sl < open) ? sl - g_dist : ask);
         RememberPosition(id, pk, 0, 0);
         k = PosIndex(id);
        }
      double peak = g_posPeak[k];
      double ns = ShieldNewSL(isBuy, open, sl, tp, bid, ask, peak, stopLvl);
      g_posPeak[k] = peak;
      if(ns > 0 && !trade.PositionModify(tk, ns, tp) && InpVerbose)
         Print("CortexML: modify failed ", trade.ResultRetcode());
     }
  }

//--- realised P/L history of this EA (for the recovery high-water mark)
void InitPnlHistory()
  {
   g_eaPnl = 0;
   g_eaHwm = 0;
   if(!HistorySelect(0, TimeCurrent()))
      return;
   int n = HistoryDealsTotal();
   for(int i = 0; i < n; i++)
     {
      ulong d = HistoryDealGetTicket(i);
      if(d == 0)
         continue;
      if(HistoryDealGetInteger(d, DEAL_MAGIC) != InpMagic || HistoryDealGetString(d, DEAL_SYMBOL) != _Symbol)
         continue;
      g_eaPnl += HistoryDealGetDouble(d, DEAL_PROFIT) + HistoryDealGetDouble(d, DEAL_COMMISSION) + HistoryDealGetDouble(d, DEAL_SWAP);
      g_eaHwm = MathMax(g_eaHwm, g_eaPnl);
     }
  }

void OnTradeTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &request, const MqlTradeResult &result)
  {
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD)
      return;
   if(!HistoryDealSelect(trans.deal))
      return;
   if(HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != InpMagic || HistoryDealGetString(trans.deal, DEAL_SYMBOL) != _Symbol)
      return;
   long entry = HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
   if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_OUT_BY)
      return;
   ulong posId = (ulong)HistoryDealGetInteger(trans.deal, DEAL_POSITION_ID);
   double vol = HistoryDealGetDouble(trans.deal, DEAL_VOLUME);
   datetime tclose = (datetime)HistoryDealGetInteger(trans.deal, DEAL_TIME);
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
   CoreOnTradeClosed(posId, money, vol, tclose);
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
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if(ask - bid > InpMaxSpreadUSD)
     {
      g_lastDecision = StringFormat("spread %.2f too wide", ask - bid);
      return;
     }
   double base = BaseLot();
   double mult = RecoveryMultiplier(base);
   double lot = NormalizeLot(base * mult);
   bool ok;
   if(sig.dir > 0)
      ok = trade.Buy(lot, _Symbol, ask, NormalizeDouble(ask - InpStopLossUSD, _Digits),
                     NormalizeDouble(ask + sig.tp, _Digits), InpComment);
   else
      ok = trade.Sell(lot, _Symbol, bid, NormalizeDouble(bid + InpStopLossUSD, _Digits),
                      NormalizeDouble(bid - sig.tp, _Digits), InpComment);
   if(ok && (trade.ResultRetcode() == TRADE_RETCODE_DONE || trade.ResultRetcode() == TRADE_RETCODE_PLACED))
     {
      ulong deal = trade.ResultDeal();
      ulong posId = 0;
      if(deal > 0 && HistoryDealSelect(deal))
         posId = (ulong)HistoryDealGetInteger(deal, DEAL_POSITION_ID);
      if(posId > 0)
         RememberPosition(posId, sig.dir > 0 ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK),
                          sig.sigT, sig.dir);
      CoreOnEntry(sig, lot, mult);
     }
   else
      g_lastDecision = "order rejected: " + IntegerToString((int)trade.ResultRetcode());
  }

//+------------------------------------------------------------------+
//| Expert lifecycle                                                 |
//+------------------------------------------------------------------+
int OnInit()
  {
   g_shiftZero = (InpServerGMTWinter == 2 && InpServerUsDst);
   if(!CoreSetupShield())
      return(INIT_PARAMETERS_INCORRECT);
   trade.SetExpertMagicNumber((ulong)InpMagic);
   trade.SetDeviationInPoints(50);
   trade.SetTypeFillingBySymbol(_Symbol);
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
   //--- custom criterion: profit factor weighted by the number of trades and drawdown
   double pf = TesterStatistics(STAT_PROFIT_FACTOR);
   double n  = TesterStatistics(STAT_TRADES);
   double dd = TesterStatistics(STAT_EQUITY_DDREL_PERCENT);
   if(n < 30)
      return(0);
   return(pf * MathSqrt(n) / MathMax(1.0, dd));
  }
//+------------------------------------------------------------------+
