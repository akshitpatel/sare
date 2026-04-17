#!/usr/bin/env bash
# run.sh — SARE-HX process manager
#
# Usage:
#   ./run.sh start           # start all 3 daemons
#   ./run.sh stop            # stop all 3 daemons
#   ./run.sh restart         # stop + start all
#   ./run.sh status          # show running state + live metrics
#   ./run.sh web             # start web server only
#   ./run.sh daemon          # start learn daemon only
#   ./run.sh evolver         # start evolver daemon only
#   ./run.sh logs [web|daemon|evolver]   # tail a log file
#   ./run.sh install         # install launchd agents (auto-start on login)
#   ./run.sh uninstall       # remove launchd agents

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="/opt/homebrew/bin/python3"
PYTHON_DIR="$ROOT/python"

# Process commands
WEB_CMD="$PYTHON_BIN -m sare.interface.web --port 8080"
DAEMON_CMD="$PYTHON_BIN $ROOT/learn_daemon.py --verbose --interval 1 --batch-size 100"
EVOLVER_CMD="$PYTHON_BIN $ROOT/evolver_daemon.py --interval 20 --budget 2.0"

# Log files
WEB_LOG="/tmp/sare_web.log"
DAEMON_LOG="/tmp/sare_daemon.log"
EVOLVER_LOG="/tmp/sare_evolver.log"

# PID files
WEB_PID="$ROOT/.pid_web"
DAEMON_PID="$ROOT/.pid_daemon"
EVOLVER_PID="$ROOT/.pid_evolver"

# ── Colors ─────────────────────────────────────────────────────────────────────
RESET="\033[0m"
BOLD="\033[1m"
GREEN="\033[92m"
YELLOW="\033[93m"
RED="\033[91m"
CYAN="\033[96m"
DIM="\033[2m"

# ── Helpers ────────────────────────────────────────────────────────────────────
_is_running() {
  local pidfile="$1"
  [ -f "$pidfile" ] || return 1
  local pid
  pid=$(cat "$pidfile")
  kill -0 "$pid" 2>/dev/null
}

_stop_pid() {
  local pidfile="$1"
  local label="$2"
  if ! _is_running "$pidfile"; then
    printf "  ${DIM}%s: not running${RESET}\n" "$label"
    rm -f "$pidfile"
    return 0
  fi
  local pid
  pid=$(cat "$pidfile")
  kill "$pid" 2>/dev/null || true
  local i=0
  while kill -0 "$pid" 2>/dev/null && [ $i -lt 10 ]; do
    sleep 0.5; i=$((i+1))
  done
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$pidfile"
  printf "  ${RED}✗${RESET} ${BOLD}%s${RESET} stopped (was pid=%s)\n" "$label" "$pid"
}

_start_proc() {
  local label="$1"
  local cmd="$2"
  local log="$3"
  local pidfile="$4"

  if _is_running "$pidfile"; then
    printf "  ${YELLOW}%s${RESET}: already running (pid %s)\n" "$label" "$(cat "$pidfile")"
    return 0
  fi

  # Rotate log if > 5000 lines
  if [ -f "$log" ] && [ "$(wc -l < "$log")" -gt 5000 ]; then
    tail -5000 "$log" > "$log.tmp" && mv "$log.tmp" "$log"
  fi

  # Start process
  (cd "$PYTHON_DIR" && nohup $cmd >> "$log" 2>&1 &
   echo $! > "$pidfile")
  sleep 0.4
  local pid
  pid=$(cat "$pidfile" 2>/dev/null || echo "?")
  printf "  ${GREEN}✓${RESET} ${BOLD}%s${RESET} started  pid=%s  log=%s\n" "$label" "$pid" "$log"
}

_status_line() {
  local label="$1"
  local pidfile="$2"
  local log="$3"
  if _is_running "$pidfile"; then
    local pid
    pid=$(cat "$pidfile")
    local stats
    stats=$(ps -o %cpu=,rss= -p "$pid" 2>/dev/null | awk '{printf "cpu=%.1f%% mem=%dMB",$1,$2/1024}' || echo "")
    printf "  ${GREEN}●${RESET} ${BOLD}%-10s${RESET}  pid=%-6s  %s\n" "$label" "$pid" "$stats"
  else
    printf "  ${RED}○${RESET} ${BOLD}%-10s${RESET}  stopped\n" "$label"
  fi
}

# ── Commands ───────────────────────────────────────────────────────────────────

cmd_start() {
  printf "\n${CYAN}${BOLD}Starting SARE-HX daemons...${RESET}\n"
  _start_proc "web"     "$WEB_CMD"     "$WEB_LOG"     "$WEB_PID"
  sleep 2  # give web server time to bind port before daemon tries to connect
  _start_proc "daemon"  "$DAEMON_CMD"  "$DAEMON_LOG"  "$DAEMON_PID"
  printf "\n  ${DIM}Evolver disabled (use './run.sh evolver' to start manually)${RESET}\n"
  printf "\n  ${DIM}Use './run.sh status' for live metrics${RESET}\n\n"
}

cmd_stop() {
  printf "\n${CYAN}${BOLD}Stopping SARE-HX daemons...${RESET}\n"
  _stop_pid "$EVOLVER_PID" "evolver"
  _stop_pid "$DAEMON_PID"  "daemon"
  _stop_pid "$WEB_PID"     "web"
  printf "\n"
}

cmd_restart() {
  cmd_stop
  sleep 2
  cmd_start
}

cmd_watch() {
  local interval="${2:-30}"
  printf "\n${CYAN}${BOLD}Watchdog active — checking every ${interval}s (Ctrl-C to stop)${RESET}\n\n"
  while true; do
    local restarted=0
    for pair in "daemon:$DAEMON_PID:$DAEMON_CMD:$DAEMON_LOG" "web:$WEB_PID:$WEB_CMD:$WEB_LOG"; do
      local name="${pair%%:*}"; local rest="${pair#*:}"
      local pidfile="${rest%%:*}"; local rest2="${rest#*:}"
      local cmd="${rest2%%:*}"; local log="${rest2#*:}"
      if ! _is_running "$pidfile"; then
        printf "  ${YELLOW}⚠${RESET}  ${BOLD}%s${RESET} died — restarting…\n" "$name"
        _start_proc "$name" "$cmd" "$log" "$pidfile"
        restarted=$((restarted+1))
      fi
    done
    [ $restarted -eq 0 ] && printf "  ${DIM}[$(date '+%H:%M:%S')] all 3 daemons alive${RESET}\r"
    sleep "$interval"
  done
}

cmd_status() {
  printf "\n${CYAN}${BOLD}══════════════════════════════════════${RESET}\n"
  printf "${CYAN}${BOLD}  SARE-HX STATUS${RESET}\n"
  printf "${CYAN}${BOLD}══════════════════════════════════════${RESET}\n"
  _status_line "web"     "$WEB_PID"     "$WEB_LOG"
  _status_line "daemon"  "$DAEMON_PID"  "$DAEMON_LOG"
  _status_line "evolver" "$EVOLVER_PID" "$EVOLVER_LOG"
  printf "\n"

  # Knowledge base quick stats (read directly from data file — no API needed)
  local wm_file="$ROOT/data/memory/world_model_v2.json"
  if [ -f "$wm_file" ]; then
    "$PYTHON_BIN" -c "
import json, sys
try:
    d = json.loads(open('$wm_file').read())
    facts = d.get('facts', {})
    total_facts = sum(len(v) for v in facts.values())
    total_links = len(d.get('causal_links', {}))
    print(f'  Knowledge: {total_facts:,} facts  |  {total_links:,} causal links  |  {len(facts)} domains')
except Exception as e:
    pass
" 2>/dev/null || true
  fi

  # Live metrics from web API
  if _is_running "$WEB_PID"; then
    local evo
    evo=$(curl -s --max-time 4 http://localhost:8080/api/agi/evolution 2>/dev/null || echo "")
    if [ -n "$evo" ]; then
      local vel
      vel=$("$PYTHON_BIN" -c "
import json,sys
d=json.loads('''$evo''')
print(f'{round(d[\"velocity_score\"]*100)}%')
" 2>/dev/null || echo "?")
      printf "  ${BOLD}AGI Velocity${RESET}: ${GREEN}%s${RESET}\n" "$vel"
      "$PYTHON_BIN" -c "
import json
d=json.loads(r'''$evo''')
for s in d.get('subsystems',[]):
  icon='✅' if s['status']=='healthy' else ('⚠️ ' if s['status']=='degraded' else '❌')
  print(f'  {icon} {s[\"name\"]:22} {s[\"status\"]}')
" 2>/dev/null || true
      printf "\n"
    fi

    local hs
    hs=$(curl -s --max-time 2 http://localhost:8080/api/homeostasis 2>/dev/null || echo "")
    if [ -n "$hs" ]; then
      printf "  ${BOLD}Homeostasis drives${RESET}:\n"
      "$PYTHON_BIN" -c "
import json
d=json.loads(r'''$hs''')
for k,v in d.get('drives',{}).items():
  pct=int(v['level']*10)
  bar='█'*pct+'░'*(10-pct)
  print(f'    {k:15} {bar} {v[\"level\"]:.2f}  [{v[\"urgency\"]}]')
" 2>/dev/null || true
      printf "\n"
    fi
  fi
}

cmd_logs() {
  local name="${1:-web}"
  local log
  case "$name" in
    web)     log="$WEB_LOG" ;;
    daemon)  log="$DAEMON_LOG" ;;
    evolver) log="$EVOLVER_LOG" ;;
    *)
      printf "Unknown: %s  (choose web|daemon|evolver)\n" "$name"
      exit 1
      ;;
  esac
  printf "${CYAN}Tailing %s${RESET}  (Ctrl-C to stop)\n" "$log"
  tail -f "$log"
}

cmd_install() {
  printf "\n${CYAN}${BOLD}Installing launchd agents (auto-start on login)...${RESET}\n"
  local plist_dir="$HOME/Library/LaunchAgents"
  mkdir -p "$plist_dir"

  _write_plist() {
    local name="$1"
    local prog="$2"
    local args="$3"    # space-separated additional args
    local log="$4"
    local plist="$plist_dir/com.sare.${name}.plist"

    # Build <string> entries for extra args
    local arg_xml=""
    for a in $args; do
      arg_xml="${arg_xml}    <string>${a}</string>\n"
    done

    cat > "$plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>             <string>com.sare.${name}</string>
  <key>ProgramArguments</key>
  <array>
    <string>${prog}</string>
$(printf '%b' "$arg_xml")  </array>
  <key>WorkingDirectory</key>  <string>${PYTHON_DIR}</string>
  <key>StandardOutPath</key>   <string>${log}</string>
  <key>StandardErrorPath</key> <string>${log}</string>
  <key>RunAtLoad</key>         <true/>
  <key>KeepAlive</key>         <true/>
  <key>ThrottleInterval</key>  <integer>10</integer>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHONPATH</key>      <string>${PYTHON_DIR}</string>
  </dict>
</dict>
</plist>
PLIST
    # Unload first (ignore errors), then load
    launchctl unload "$plist" 2>/dev/null || true
    launchctl load   "$plist"
    printf "  ${GREEN}✓${RESET} com.sare.%s  →  %s\n" "$name" "$plist"
  }

  _write_plist "web" \
    "$PYTHON_BIN" \
    "-m sare.interface.web --port 8080" \
    "$WEB_LOG"

  _write_plist "daemon" \
    "$PYTHON_BIN" \
    "$ROOT/learn_daemon.py --verbose --interval 5 --batch-size 50" \
    "$DAEMON_LOG"

  printf "\n  ${DIM}Evolver not installed (disabled — start manually with ./run.sh evolver).${RESET}\n"
  printf "\n  ${DIM}Both agents auto-start on login and restart on crash.${RESET}\n"
  printf "  ${DIM}Remove with: ./run.sh uninstall${RESET}\n\n"
}

cmd_uninstall() {
  printf "\n${CYAN}${BOLD}Removing launchd agents...${RESET}\n"
  local plist_dir="$HOME/Library/LaunchAgents"
  for name in web daemon evolver; do
    local plist="$plist_dir/com.sare.${name}.plist"
    launchctl unload "$plist" 2>/dev/null || true
    rm -f "$plist"
    printf "  ${RED}✗${RESET} removed com.sare.%s\n" "$name"
  done
  printf "\n"
}

# ── Dispatch ───────────────────────────────────────────────────────────────────
CMD="${1:-status}"
case "$CMD" in
  start|all)  cmd_start ;;
  stop)       cmd_stop ;;
  restart)    cmd_restart ;;
  status)     cmd_status ;;
  watch)      cmd_watch "$@" ;;
  web)        _start_proc "web"     "$WEB_CMD"     "$WEB_LOG"     "$WEB_PID" ;;
  daemon)     _start_proc "daemon"  "$DAEMON_CMD"  "$DAEMON_LOG"  "$DAEMON_PID" ;;
  evolver)    _start_proc "evolver" "$EVOLVER_CMD" "$EVOLVER_LOG" "$EVOLVER_PID" ;;
  logs)       cmd_logs "${2:-web}" ;;
  install)    cmd_install ;;
  uninstall)  cmd_uninstall ;;
  *)
    printf "Usage: %s {start|stop|restart|status|watch|web|daemon|evolver|logs [web|daemon|evolver]|install|uninstall}\n" "$0"
    exit 1
    ;;
esac
