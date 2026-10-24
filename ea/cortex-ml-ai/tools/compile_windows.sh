#!/bin/bash
# Cortex ML AI: install + compile in every MetaTrader 4 / 5 terminal of this Windows PC (run in Git Bash).
#   bash ea/cortex-ml-ai/tools/compile_windows.sh
# For each data folder %APPDATA%\MetaQuotes\Terminal\<id> it copies the expert, the shared core and the
# presets, compiles with that terminal's own MetaEditor and prints the errors/warnings from the log.
set -u
SRC="$(cd "$(dirname "$0")/.." && pwd)"
APP="$(cygpath -u "$APPDATA")"
found=0
for org in "$APP"/MetaQuotes/Terminal/*/origin.txt; do
  [ -f "$org" ] || continue
  data="$(dirname "$org")"
  inst="$(iconv -f UTF-16LE -t UTF-8 "$org" 2>/dev/null | sed 's/^\xEF\xBB\xBF//' | tr -d '\r\n')"
  [ -n "$inst" ] || continue
  instu="$(cygpath -u "$inst")"
  if [ -d "$data/MQL4" ]; then plat=MQL4; ext=mq4; me="$instu/metaeditor.exe"; else plat=MQL5; ext=mq5; me="$instu/metaeditor64.exe"; fi
  if [ ! -f "$me" ]; then echo "skip: $inst (MetaEditor not found)"; continue; fi
  dst="$data/$plat/Experts/CortexML"
  mkdir -p "$dst" "$data/$plat/Presets"
  cp "$SRC/$plat/Experts/CortexML/"* "$dst/"
  cp "$SRC/$plat/Presets/"*.set "$data/$plat/Presets/" 2>/dev/null
  log="$dst/compile.log"
  rm -f "$log"
  "$me" /compile:"$(cygpath -w "$dst/CortexML_AI.$ext")" /log:"$(cygpath -w "$log")" >/dev/null 2>&1
  echo "== $inst  ($plat)"
  if [ -f "$log" ]; then
    iconv -f UTF-16LE -t UTF-8 "$log" 2>/dev/null | tr -d '\r' | grep -iE "error|warning|result" | head -80
  else
    echo "   no log produced"
  fi
  [ -f "$dst/CortexML_AI.${ext/mq/ex}" ] && echo "   OK -> $dst/CortexML_AI.${ext/mq/ex}"
  found=1
done
[ $found = 1 ] || echo "No MetaTrader data folders found under %APPDATA%\\MetaQuotes\\Terminal"
