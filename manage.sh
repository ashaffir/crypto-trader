#!/usr/bin/env bash
# Crypto Trader Development & Deployment Manager
# Interactive menu + CLI commands for dev, debug, and docker operations

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors (only if terminal supports them)
if [[ -t 1 ]] && command -v tput >/dev/null 2>&1 && tput setaf 1 >/dev/null 2>&1; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BLUE='\033[0;34m'
  CYAN='\033[0;36m'
  NC='\033[0m'
else
  RED=''
  GREEN=''
  YELLOW=''
  BLUE=''
  CYAN=''
  NC=''
fi

# Print functions
info()    { printf "${CYAN}ℹ${NC}  %s\n" "$*"; }
ok()      { printf "${GREEN}✓${NC} %s\n" "$*"; }
warn()    { printf "${YELLOW}⚠${NC}  %s\n" "$*"; }
error()   { printf "${RED}✗${NC} %s\n" "$*" >&2; }
die()     { error "$*"; exit 1; }
pause()   { echo; read -p "Press Enter to continue..." -r; echo; }

# Check if command exists
have() { command -v "$1" >/dev/null 2>&1; }

# Detect docker compose command
detect_compose() {
  if have docker && docker compose version &>/dev/null; then
    echo "docker compose"
  elif have docker-compose; then
    echo "docker-compose"
  else
    die "Docker Compose not found"
  fi
}

DC=$(detect_compose 2>/dev/null || echo "")

#
# Development Commands
#

cmd_dev_ui() {
  local port="${1:-8501}"
  info "Starting Streamlit UI on port ${port}..."
  printf "${GREEN}→${NC} Open: http://localhost:${port}\n"
  export PYTHONPATH="$SCRIPT_DIR"
  exec streamlit run ui/Home.py \
    --server.port="$port" \
    --server.address=0.0.0.0 \
    --server.runOnSave=true
}

cmd_dev_bot() {
  info "Starting trading bot locally..."
  export PYTHONPATH="$SCRIPT_DIR"
  exec python3 -m src.supervisor "$@"
}

cmd_dev_test() {
  local pattern="${1:-}"
  if ! have pytest; then
    die "pytest not found. Run: pip install -r requirements.txt"
  fi
  info "Running tests..."
  export PYTHONPATH="$SCRIPT_DIR"
  if [[ -n "$pattern" ]]; then
    pytest -xvs -k "$pattern" tests/
  else
    pytest -xvs tests/
  fi
}

cmd_dev_shell() {
  info "Starting Python shell..."
  export PYTHONPATH="$SCRIPT_DIR"
  python3 <<'PYEOF'
import sys
sys.path.insert(0, '.')
from src.config import load_config
from src.utils.data_window import construct_data_window
from src.utils.llm_client import LLMClient, LLMConfig
from ui.lib.settings_state import *
import pandas as pd
import json

print("Loaded: config, data_window, llm_client, settings_state, pandas, json")
print()
import code
code.interact(local=locals())
PYEOF
}

#
# Debug Commands
#

cmd_debug_data() {
  info "Logbook data summary:"
  echo
  
  local logbook="$SCRIPT_DIR/data/logbook"
  if [[ ! -d "$logbook" ]]; then
    warn "No logbook directory: $logbook"
    return
  fi
  
  for table in market_snapshot trade_recommendation; do
    local path="$logbook/$table"
    if [[ -d "$path" ]]; then
      local symbols=$(find "$path" -type d -name "symbol=*" 2>/dev/null | sed 's/.*symbol=//' | sort -u | tr '\n' ' ')
      local file_count=$(find "$path" -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')
      local size=$(du -sh "$path" 2>/dev/null | cut -f1)
      
      echo -e "${CYAN}$table${NC}"
      echo "  Symbols: ${symbols:-none}"
      echo "  Files:   $file_count"
      echo "  Size:    $size"
      echo
    fi
  done
}

cmd_debug_config() {
  info "Configuration files:"
  echo
  for file in configs/config.yaml data/control/runtime_config.json data/control/llm_configs.json; do
    if [[ -f "$file" ]]; then
      echo -e "${GREEN}✓${NC} $file"
      if [[ "$file" == *.json ]]; then
        python3 -c "import json; print(json.dumps(json.load(open('$file')), indent=2))" 2>/dev/null || cat "$file"
      else
        cat "$file"
      fi
      echo
    else
      echo -e "${YELLOW}⚠${NC} $file (not found)"
    fi
  done
}

cmd_debug_symbols() {
  info "Tracked symbols:"
  export PYTHONPATH="$SCRIPT_DIR"
  python3 <<'PYEOF'
from ui.lib.settings_state import load_tracked_symbols
symbols = load_tracked_symbols()
if symbols:
    for s in symbols:
        print(f"  • {s}")
else:
    print("  (none)")
PYEOF
}

cmd_debug_llm() {
  info "LLM configuration:"
  export PYTHONPATH="$SCRIPT_DIR"
  python3 <<'PYEOF'
from ui.lib.settings_state import load_llm_configs, load_window_seconds
configs = load_llm_configs()
window = load_window_seconds()
print(f"\nWindow Size: {window} seconds\n")
print(f"Configured LLMs: {len(configs)}\n")
for cfg in configs:
    active = "✓ ACTIVE" if cfg.get("is_active") else ""
    print(f"  • {cfg.get('name')} {active}")
    print(f"    Provider: {cfg.get('provider')}")
    print(f"    Model: {cfg.get('model')}")
    print(f"    Endpoint: {cfg.get('base_url')}")
    print(f"    API Key: {'*' * 20 if cfg.get('api_key') else '(none)'}")
PYEOF
}

cmd_debug_window() {
  local seconds="${1:-60}"
  info "Testing DATA_WINDOW (${seconds}s)..."
  export PYTHONPATH="$SCRIPT_DIR"
  python3 <<PYEOF
import json
from src.utils.data_window import construct_data_window
from ui.lib.settings_state import load_tracked_symbols
symbols = load_tracked_symbols()
print(f"Symbols: {symbols}")
try:
    data = construct_data_window("data/logbook", symbols, ${seconds})
    print(f"\nTimestamp: {data['timestamp']}")
    print(f"Window: {data['window_seconds']}s")
    print(f"Assets: {len(data['assets'])}\n")
    for asset in data['assets']:
        print(f"  {asset['symbol']}:")
        print(f"    Samples:      {len(asset['recent_prices'])} prices")
        print(f"    Price change: {asset['price_change_bps']:+.2f} bps")
        print(f"    Volume total: {asset['volume_total']:.2f}")
        if asset['recent_prices']:
            print(f"    Price range:  {asset['recent_prices'][0]:.2f} → {asset['recent_prices'][-1]:.2f}")
        print()
    print(f"JSON size: {len(json.dumps(data))} bytes")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
PYEOF
}

cmd_debug_parquet() {
  export PYTHONPATH="$SCRIPT_DIR"
  python3 src/parquet_inspector.py "$@"
}

#
# Docker Commands
#

cmd_up() {
  [[ -z "$DC" ]] && die "Docker Compose not available"
  if [[ "${1:-}" == "fg" ]]; then
    info "Starting services (foreground)..."
    $DC up
  else
    info "Starting services (background)..."
    $DC up -d
    ok "Services started"
  fi
}

cmd_down() {
  [[ -z "$DC" ]] && die "Docker Compose not available"
  info "Stopping services..."
  $DC down
  ok "Services stopped"
}

cmd_restart() {
  [[ -z "$DC" ]] && die "Docker Compose not available"
  local svc="${1:-}"
  if [[ -n "$svc" ]]; then
    info "Restarting $svc..."
    $DC restart "$svc"
  else
    info "Restarting all services..."
    $DC restart
  fi
  ok "Restarted"
}

cmd_logs() {
  [[ -z "$DC" ]] && die "Docker Compose not available"
  local svc="${1:-}"
  local follow="${2:-}"
  if [[ "$follow" == "-f" || "$follow" == "--follow" ]]; then
    $DC logs -f --tail=100 $svc
  else
    $DC logs --tail=100 $svc
  fi
}

cmd_build() {
  [[ -z "$DC" ]] && die "Docker Compose not available"
  info "Building images..."
  $DC build --no-cache
  ok "Build complete"
}

cmd_build_restart() {
  [[ -z "$DC" ]] && die "Docker Compose not available"
  info "Building images..."
  $DC build --no-cache
  ok "Build complete"
  echo
  info "Stopping services..."
  $DC down
  ok "Services stopped"
  echo
  info "Starting services..."
  $DC up -d
  ok "Services started"
}

cmd_ps() {
  [[ -z "$DC" ]] && die "Docker Compose not available"
  $DC ps
}

#
# Maintenance
#

cmd_clean() {
  info "Cleaning Python cache..."
  find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
  find . -type f -name "*.pyc" -delete 2>/dev/null || true
  find . -type f -name "*.pyo" -delete 2>/dev/null || true
  ok "Cache cleaned"
}

cmd_doctor() {
  info "Running diagnostics...\n"
  have python3 && ok "python3: $(python3 --version)" || error "python3 not found"
  python3 -c "import streamlit" 2>/dev/null && ok "streamlit installed" || error "streamlit not installed"
  python3 -c "import pytest" 2>/dev/null && ok "pytest installed" || warn "pytest not installed"
  have docker && ok "docker: $(docker --version)" || warn "docker not found"
  [[ -n "$DC" ]] && ok "docker compose available" || warn "docker compose not available"
  
  for dir in data/logbook data/control configs; do
    [[ -d "$dir" ]] && ok "directory: $dir" || error "missing: $dir"
  done
  
  for file in configs/config.yaml requirements.txt; do
    [[ -f "$file" ]] && ok "file: $file" || error "missing: $file"
  done
  
  echo
  info "Diagnostics complete"
}

#
# Interactive Menu
#

show_header() {
  clear
  printf "${BLUE}╔══════════════════════════════════════════════╗${NC}\n"
  printf "${BLUE}║      Crypto Trader - Manager v2.0           ║${NC}\n"
  printf "${BLUE}╚══════════════════════════════════════════════╝${NC}\n"
  echo
}

docker_management_menu() {
  while true; do
    show_header
    printf "${BLUE}🐋 Docker Management${NC}\n"
    echo "====================="
    echo
    
    printf "${CYAN}Current Status:${NC}\n"
    if [[ -n "$DC" ]] && $DC ps 2>/dev/null | grep -q "Up"; then
      printf "  ${GREEN}✓${NC} Services running\n"
      $DC ps
    else
      printf "  ${YELLOW}○${NC} Services stopped\n"
    fi
    echo
    
    printf "${YELLOW}📋 Available Actions:${NC}\n"
    echo "======================"
    printf "  ${CYAN}1)${NC} 🚀 Start services\n"
    printf "  ${CYAN}2)${NC} 🛑 Stop services\n"
    printf "  ${CYAN}3)${NC} 🔄 Restart services\n"
    printf "  ${CYAN}4)${NC} 📋 View logs\n"
    printf "  ${CYAN}5)${NC} 🔨 Build images\n"
    printf "  ${CYAN}6)${NC} 🔨🔄 Build & Restart\n"
    printf "  ${CYAN}7)${NC} 📊 Show containers\n"
    printf "  ${CYAN}b)${NC} ⬅️  Back to main menu\n"
    echo
    
    printf "${CYAN}Choice:${NC} "
    read -r choice
    echo
    
    case "$choice" in
      1) cmd_up; pause ;;
      2) cmd_down; pause ;;
      3) cmd_restart; pause ;;
      4)
        read -p "Service name (or leave empty for all): " svc
        cmd_logs "$svc" "--follow"
        ;;
      5) cmd_build; pause ;;
      6) cmd_build_restart; pause ;;
      7) cmd_ps; pause ;;
      b|B) return ;;
      *) error "Invalid choice"; sleep 1 ;;
    esac
  done
}

show_status() {
  printf "${BLUE}📊 Status:${NC}\n"
  echo "=========="
  
  # Check if services are running
  if [[ -n "$DC" ]] && $DC ps 2>/dev/null | grep -q "Up"; then
    printf "  ${GREEN}✓${NC} Docker services running\n"
  else
    printf "  ${YELLOW}○${NC} Docker services stopped\n"
  fi
  
  # Check data
  local logbook="data/logbook/market_snapshot"
  if [[ -d "$logbook" ]]; then
    local files=$(find "$logbook" -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')
    printf "  ${GREEN}✓${NC} Data: $files parquet files\n"
  else
    printf "  ${YELLOW}○${NC} No data collected yet\n"
  fi
  
  echo
}

show_menu() {
  show_header
  show_status
  
  printf "${GREEN}Development:${NC}\n"
  printf "  ${CYAN}1)${NC} Start UI\n"
  printf "  ${CYAN}2)${NC} Start Bot\n"
  printf "  ${CYAN}3)${NC} Run Tests\n"
  printf "  ${CYAN}4)${NC} Python Shell\n"
  echo
  printf "${GREEN}Debug:${NC}\n"
  printf "  ${CYAN}5)${NC} Show Data Summary\n"
  printf "  ${CYAN}6)${NC} Show Config\n"
  printf "  ${CYAN}7)${NC} Show LLM Config\n"
  printf "  ${CYAN}8)${NC} Test Data Window\n"
  printf "  ${CYAN}9)${NC} Inspect Parquet\n"
  echo
  printf "${GREEN}Docker:${NC}\n"
  printf "  ${CYAN}10)${NC} 🐋 Docker Management\n"
  echo
  printf "${GREEN}Other:${NC}\n"
  printf "  ${CYAN}11)${NC} Clean Cache\n"
  printf "  ${CYAN}12)${NC} Diagnostics\n"
  printf "  ${CYAN}h)${NC}  Help / CLI Usage\n"
  printf "  ${CYAN}q)${NC}  Quit\n"
  echo
}

interactive_menu() {
  while true; do
    show_menu
    printf "${CYAN}Choice:${NC} "
    read -r choice
    echo
    
    case "$choice" in
      1) cmd_dev_ui ;;
      2) cmd_dev_bot ;;
      3) cmd_dev_test; pause ;;
      4) cmd_dev_shell ;;
      5) cmd_debug_data; pause ;;
      6) cmd_debug_config; pause ;;
      7) cmd_debug_llm; pause ;;
      8)
        read -p "Window size in seconds [60]: " secs
        cmd_debug_window "${secs:-60}"
        pause
        ;;
      9)
        read -p "Command (e.g., 'list' or 'head market_snapshot BTCUSDT'): " args
        cmd_debug_parquet $args
        pause
        ;;
      10) docker_management_menu ;;
      11) cmd_clean; pause ;;
      12) cmd_doctor; pause ;;
      h|H) show_cli_help; pause ;;
      q|Q) info "Goodbye!"; exit 0 ;;
      *) error "Invalid choice"; sleep 1 ;;
    esac
  done
}

show_cli_help() {
  printf "${CYAN}CLI Usage:${NC} ./manage.sh <command> [options]\n\n"
  
  printf "${GREEN}Development:${NC}\n"
  echo "  dev ui [--port 8501]    Start Streamlit UI"
  echo "  dev bot [opts]          Start trading bot"
  echo "  dev test [pattern]      Run tests"
  echo "  dev shell               Python shell"
  echo
  
  printf "${GREEN}Debug:${NC}\n"
  echo "  debug data              Show data summary"
  echo "  debug config            Show config files"
  echo "  debug symbols           Show tracked symbols"
  echo "  debug llm               Show LLM config"
  echo "  debug window [secs]     Test DATA_WINDOW"
  echo "  debug parquet <args>    Inspect parquet files"
  echo
  
  printf "${GREEN}Docker:${NC}\n"
  echo "  up [fg]                 Start services"
  echo "  down                    Stop services"
  echo "  restart [service]       Restart service(s)"
  echo "  logs [service] [-f]     View logs"
  echo "  build                   Build images"
  echo "  build-restart           Build & restart"
  echo "  ps                      Show containers"
  echo
  
  printf "${GREEN}Maintenance:${NC}\n"
  echo "  clean                   Clean cache"
  echo "  doctor                  Run diagnostics"
  echo
  
  printf "${GREEN}Examples:${NC}\n"
  echo "  ./manage.sh dev ui --port 8502"
  echo "  ./manage.sh debug window 120"
  echo "  ./manage.sh logs bot --follow"
  echo "  ./manage.sh                    # Interactive menu"
  echo
}

#
# Main
#

main() {
  # No arguments = interactive menu
  if [[ $# -eq 0 ]]; then
    interactive_menu
    exit 0
  fi
  
  # Parse CLI commands
  local cmd="$1"
  shift || true
  
  case "$cmd" in
    # Development
    dev)
      local subcmd="${1:-}"
      shift || true
      case "$subcmd" in
        ui) cmd_dev_ui "$@" ;;
        bot) cmd_dev_bot "$@" ;;
        test) cmd_dev_test "$@" ;;
        shell) cmd_dev_shell ;;
        *) die "Unknown dev command: $subcmd" ;;
      esac
      ;;
    
    # Debug
    debug)
      local subcmd="${1:-}"
      shift || true
      case "$subcmd" in
        data) cmd_debug_data ;;
        config) cmd_debug_config ;;
        symbols) cmd_debug_symbols ;;
        llm) cmd_debug_llm ;;
        window) cmd_debug_window "$@" ;;
        parquet) cmd_debug_parquet "$@" ;;
        *) die "Unknown debug command: $subcmd" ;;
      esac
      ;;
    
    # Docker
    up) cmd_up "$@" ;;
    down) cmd_down ;;
    restart) cmd_restart "$@" ;;
    logs) cmd_logs "$@" ;;
    build) cmd_build ;;
    build-restart) cmd_build_restart ;;
    ps) cmd_ps ;;
    
    # Maintenance
    clean) cmd_clean ;;
    doctor) cmd_doctor ;;
    
    # Help
    -h|--help|help) show_cli_help ;;
    
    *)
      error "Unknown command: $cmd"
      echo
      show_cli_help
      exit 1
      ;;
  esac
}

main "$@"
