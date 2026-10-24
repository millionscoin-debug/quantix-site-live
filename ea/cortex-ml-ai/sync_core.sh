#!/bin/sh
# CortexCore.mqh is shared 1:1 by the MT5 and MT4 experts: edit core/CortexCore.mqh, then run this.
cd "$(dirname "$0")" && cp core/CortexCore.mqh MQL5/Experts/CortexML/ && cp core/CortexCore.mqh MQL4/Experts/CortexML/ && md5sum core/CortexCore.mqh MQL*/Experts/CortexML/CortexCore.mqh
