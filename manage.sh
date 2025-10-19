#!/usr/bin/env bash

# Interactive project manager for development and deployment.
# - Works with Docker Compose (prefers `docker compose`, falls back to `docker-compose`).
# - Interactive menu (default), plus non-interactive subcommands for CI and scripts.
# - Adds `doctor` (env diagnostics) and `dev` (local runs without Docker).
# - Still supports flags: --help, --dry-run, --self-test, --file, --inspect, --prune.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
COMPOSE_FILE="${DEFAULT_COMPOSE_FILE}"
DRY_RUN="false"
FORCE_DOCKER_TOOL=""  # internal/testing override: docker|docker-compose
CMD=""                 # subcommand, if provided
CMD_ARGS=()            # subcommand args

color() {
  local code="$1"; shift || true
  if command -v tput >/dev/null 2>&1; then
    tput setaf "$code" || true
  fi
}

reset_color() {
  if command -v tput >/dev/null 2>&1; then
    tput sgr0 || true
  fi
}

notice() { printf "%s\n" "$*"; }
info()   { printf "%s%s%s\n" "$(color 6)" "$*" "$(reset_color)"; }
warn()   { printf "%s%s%s\n" "$(color 3)" "$*" "$(reset_color)"; }
error()  { printf "%s%s%s\n" "$(color 1)" "$*" "$(reset_color)" 1>&2; }

die() { error "Error: $*"; exit 1; }

spinner_start() {
  # no-op simple placeholder to avoid external deps; kept for future enhancement
  :
}

spinner_stop() {
  :
}

run_cmd() {
  # Runs a command; when DRY_RUN=true, prints instead of executing
  if [[ "$DRY_RUN" == "true" ]]; then
    printf "DRY-RUN: %q" "$1"
    shift || true
    while (($#)); do printf " %q" "$1"; shift; done
    printf "\n"
    return 0
  fi
  "$@"
}

have() { command -v "$1" >/dev/null 2>&1; }

detect_dc() {
  # Determine docker compose CLI (prefer modern 'docker compose')
  local dc_cmd=""
  if [[ -n "$FORCE_DOCKER_TOOL" ]]; then
    case "$FORCE_DOCKER_TOOL" in
      docker)         dc_cmd="docker compose" ;;
      docker-compose) dc_cmd="docker-compose" ;;
      *) die "Unknown FORCE_DOCKER_TOOL=$FORCE_DOCKER_TOOL" ;;
    esac
  else
    # Prefer legacy docker-compose first for environments without plugin
    if have docker-compose && docker-compose version >/dev/null 2>&1; then
      dc_cmd="docker-compose"
    elif have docker && docker compose version >/dev/null 2>&1; then
      dc_cmd="docker compose"
    elif have docker && docker --help 2>&1 | grep -q "compose"; then
      # Older docker may not support 'docker compose version' but has plugin
      dc_cmd="docker compose"
    else
      dc_cmd=""
    fi
  fi

  if [[ -z "$dc_cmd" ]]; then
    warn "Neither 'docker compose' nor 'docker-compose' found."
    warn "Menu will still work in --dry-run mode, but real actions require Docker."
  fi

  printf "%s" "$dc_cmd"
}

DC_BASE=""  # set in main

dc() {
  # Wrapper to run docker compose with our -f file
  if [[ -z "$DC_BASE" ]]; then
    DC_BASE="$(detect_dc)"
  fi
  if [[ -z "$DC_BASE" ]]; then
    # Allow running in dry-run without docker installed
    if [[ "$DRY_RUN" == "true" ]]; then
      run_cmd echo "<docker-compose> -f" "$COMPOSE_FILE" "$@"
      return 0
    fi
    die "Docker Compose CLI not found. Install Docker Desktop or docker-compose."
  fi

  run_cmd $DC_BASE -f "$COMPOSE_FILE" "$@"
}

list_services() {
  # Prefer compose to list services; if unavailable, attempt to parse YAML minimally.
  if [[ -n "$DC_BASE" ]] || have docker || have docker-compose; then
    # 1) Use our wrapper
    if services_out=$(dc config --services 2>/dev/null); then
      if [[ -n "$services_out" ]]; then
        printf "%s\n" "$services_out"
        return 0
      fi
    fi
    # 2) Try legacy docker-compose explicitly
    if have docker-compose; then
      if services_out=$(docker-compose -f "$COMPOSE_FILE" config --services 2>/dev/null); then
        if [[ -n "$services_out" ]]; then
          printf "%s\n" "$services_out"
          return 0
        fi
      fi
    fi
    # 3) Try docker compose plugin explicitly
    if have docker; then
      if services_out=$(docker compose -f "$COMPOSE_FILE" config --services 2>/dev/null); then
        if [[ -n "$services_out" ]]; then
          printf "%s\n" "$services_out"
          return 0
        fi
      fi
    fi
  fi
  # Fallback: naive YAML parse of service names under top-level 'services:'
  # Note: this is a simple heuristic; adequate for local self-test without docker.
  awk '
    $0 ~ /^services:/ {in_services=1; next}
    # two leading spaces then key:
    in_services && match($0, /^[[:space:]][[:space:]]([A-Za-z0-9_.-]+):/, m) {print m[1]}
    in_services && NF==0 {in_services=0}
  ' "$COMPOSE_FILE" 2>/dev/null || true
}

confirm() {
  local prompt=${1:-"Are you sure?"}
  local default=${2:-"n"}
  local ans
  if [[ "$default" == "y" ]]; then
    read -r -p "${prompt} [Y/n] " ans || true
    ans=${ans:-Y}
  else
    read -r -p "${prompt} [y/N] " ans || true
    ans=${ans:-N}
  fi
  [[ "$ans" == "y" || "$ans" == "Y" ]]
}

choose_service() {
  local services=()
  while IFS= read -r svc; do
    [[ -n "$svc" ]] && services+=("$svc")
  done < <(list_services)
  if ((${#services[@]}==0)); then
    die "No services found in compose file: $COMPOSE_FILE"
  fi
  (
    # Subshell: redirect menu and prompts to stderr; print selection to stdout
    exec 3>&1
    exec 1>&2
    printf "Select a service:\n"
    select svc in "${services[@]}"; do
      if [[ -n "${svc:-}" ]]; then
        printf "%s" "$svc" >&3
        break
      fi
      printf "Invalid selection.\n" 1>&2
    done
  )
}

choose_shell() {
  local choices=("bash" "sh" "python" "custom")
  notice "Select a shell/command:"
  select c in "${choices[@]}"; do
    case "$c" in
      bash|sh|python) printf "%s" "$c"; return 0 ;;
      custom)
        read -r -p "Enter custom command: " cmd
        printf "%s" "$cmd"; return 0 ;;
      *) warn "Invalid selection." ;;
    esac
  done
}

action_build()      { dc build; }
action_pull()       { dc pull; }
action_up_detached(){ dc up -d; }
action_up_fg()      { dc up; }
action_down()       { confirm "This will stop and remove containers. Proceed?" n && dc down || notice "Cancelled."; }
action_ps()         { dc ps; }
action_logs_follow(){ local s; s=$(choose_service); dc logs -f "$s"; }
action_logs_once()  { local s; s=$(choose_service); dc logs --tail=200 "$s"; }
action_restart_svc(){ local s; s=$(choose_service); dc restart "$s"; }
action_stop_svc()   { local s; s=$(choose_service); dc stop "$s"; }
action_start_all()  { action_up_detached; }
action_stop_all()   { dc stop; }
action_restart_all(){ dc restart; }
action_build_restart_all() { dc up -d --build; }
action_build_restart_svc() { local s; s=$(choose_service); dc up -d --build "$s"; }
action_exec()       {
  local s; s=$(choose_service)
  local cmd; cmd=$(choose_shell)
  # If cmd is bash/sh, run interactive tty
  case "$cmd" in
    bash|sh) dc exec -it "$s" "$cmd" ;;
    *)       dc exec "$s" sh -lc "$cmd" ;;
  esac
}
action_rebuild_nocache(){ dc build --no-cache; }
action_config()     { dc config; }
action_prune()      {
  warn "This will prune dangling images and unused volumes (not in use)."
  if confirm "Run docker image prune -f and volume prune -f?" n; then
    run_cmd docker image prune -f
    run_cmd docker volume prune -f
  else
    notice "Cancelled."
  fi
}

print_menu() {
  cat <<'EOF'
Choose an action:
  1) Build images
  2) Pull images
  3) Up (detached)
  4) Up (foreground)
  5) Down (stop & remove)
  6) Status (ps)
  7) Logs (follow)
  8) Logs (once)
  9) Restart service
 10) Stop service
 11) Exec into service
 12) Rebuild (no-cache)
 13) Show compose config
 14) Prune (dangling images/volumes)
 15) Start all
 16) Stop all
 17) Restart all
 18) Build & Restart (all)
 19) Build & Restart (service)
 20) Exit
EOF
}

interactive_menu() {
  local choice
  while true; do
    print_menu
    read -r -p "Enter choice [1-20]: " choice || true
    case "$choice" in
      1)  action_build ;;
      2)  action_pull ;;
      3)  action_up_detached ;;
      4)  action_up_fg ;;
      5)  action_down ;;
      6)  action_ps ;;
      7)  action_logs_follow ;;
      8)  action_logs_once ;;
      9)  action_restart_svc ;;
      10) action_stop_svc ;;
      11) action_exec ;;
      12) action_rebuild_nocache ;;
      13) action_config ;;
      14) action_prune ;;
      15) action_start_all ;;
      16) action_stop_all ;;
      17) action_restart_all ;;
      18) action_build_restart_all ;;
      19) action_build_restart_svc ;;
      20) notice "Bye!"; break ;;
      *)  warn "Invalid choice." ;;
    esac
  done
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -f, --file PATH     Compose file to use (default: ${COMPOSE_FILE})
      --dry-run       Print commands instead of executing
      --self-test     Run internal self-test without requiring Docker
      --print-dc      Print detected compose command and exit
      --inspect ARGS  Run parquet inspector (passes ARGS to python src/parquet_inspector.py)
      --prune  ARGS   Run data retention pruner (passes ARGS to python src/data_retention.py)
      --use-docker-compose  Force legacy docker-compose CLI
      --use-docker          Force docker compose plugin CLI
  -h, --help          Show this help

Commands (non-interactive):
  up [fg|detached]           Start all services (default: detached)
  down                       Stop and remove containers
  ps                         Show service status
  build [--no-cache]         Build images
  pull                       Pull images
  logs [svc] [follow]        Show logs (follow if specified)
  restart [svc]              Restart a service (prompt if missing)
  stop [svc]                 Stop a service (prompt if missing)
  start-all                  Start all services (alias for 'up detached')
  stop-all                   Stop all services (no removal)
  restart-all                Restart all services
  build-restart [all|svc]    Build images and restart service(s)
  exec [svc] [bash|sh|python|-- cmd]
                             Exec into service; with '--', run custom command
  config                     Show composed config
  prune-docker               Prune dangling images/volumes
  doctor                     Run environment diagnostics
  dev bot                    Run bot locally: python -m src.supervisor
  dev ui [--port 8501]       Run UI locally: streamlit run ui/ui_app.py

Without a command, an interactive menu is shown.
EOF
}

self_test() {
  notice "Running self-test in dry-run mode..."
  local prev_dry="$DRY_RUN"; DRY_RUN="true"
  FORCE_DOCKER_TOOL=""  # ensure detection runs
  DC_BASE=""            # reset

  notice "1) Detect compose command"
  local dc_detected
  dc_detected="$(detect_dc || true)"
  if [[ -z "$dc_detected" ]]; then
    notice "   docker compose not found (OK for self-test)"
  else
    notice "   detected: $dc_detected"
  fi

  notice "2) List services (fallback parse if needed)"
  local svcs
  svcs="$(list_services || true)"
  if [[ -z "$svcs" ]]; then
    warn "   No services detected in $COMPOSE_FILE"
  else
    notice "   services: $(echo "$svcs" | tr '\n' ' ')"
  fi

  notice "3) Simulate common actions"
  DC_BASE="docker compose"  # pretend for printing
  action_build
  action_up_detached
  action_ps
  DRY_RUN="$prev_dry"
  notice "Self-test complete."
}

parse_args() {
  while (($#)); do
    case "$1" in
      -f|--file)
        shift; [[ $# -gt 0 ]] || die "--file requires a path"
        COMPOSE_FILE="$1"
        ;;
      --dry-run)
        DRY_RUN="true"
        ;;
      --self-test)
        self_test; exit 0
        ;;
      --print-dc)
        DC_BASE="$(detect_dc)"; printf "%s\n" "${DC_BASE:-<not found>}"; exit 0
        ;;
      --use-docker-compose)
        FORCE_DOCKER_TOOL="docker-compose"
        ;;
      --use-docker)
        FORCE_DOCKER_TOOL="docker"
        ;;
      --inspect)
        shift || die "--inspect requires ARGS (e.g., 'list' or 'head market_snapshot BTCUSDT')"
        # pass all remaining args to inspector
        PYTHONPATH="$SCRIPT_DIR" python3 "$SCRIPT_DIR/src/parquet_inspector.py" "$@"
        exit $?
        ;;
      --prune)
        shift || die "--prune requires ARGS (e.g., '--max-days 7' or '--size-cap 50GB')"
        PYTHONPATH="$SCRIPT_DIR" python3 "$SCRIPT_DIR/src/data_retention.py" "$@"
        exit $?
        ;;
      -h|--help)
        usage; exit 0
        ;;
      --)
        shift || true
        ;;
      -* )
        die "Unknown argument: $1"
        ;;
      * )
        # First non-option -> treat as subcommand; rest are its args
        CMD="$1"; shift || true
        CMD_ARGS=("$@")
        return 0
        ;;
    esac
    shift || true
  done
}

doctor() {
  notice "Running diagnostics..."
  # Docker and Compose
  local docker_ok="no" compose_cmd
  if have docker; then
    docker_ok="yes"
    notice "docker: $(docker --version 2>/dev/null || echo not found)"
  else
    warn "docker not found"
  fi
  compose_cmd="$(detect_dc || true)"
  if [[ -n "$compose_cmd" ]]; then
    notice "compose: $compose_cmd"
    run_cmd $compose_cmd version || true
  else
    warn "docker compose/compose plugin not available"
  fi

  # Python & Streamlit (for local dev)
  if have python3; then
    notice "python3: $(python3 -V 2>/dev/null)"
  else
    warn "python3 not found"
  fi
  if have streamlit; then
    notice "streamlit: $(streamlit --version 2>/dev/null)"
  else
    warn "streamlit CLI not found"
  fi

  # Paths and permissions
  local paths=("$SCRIPT_DIR/configs" "$SCRIPT_DIR/data/logbook" "$SCRIPT_DIR/data/control" "$COMPOSE_FILE")
  for p in "${paths[@]}"; do
    if [[ -e "$p" ]]; then
      notice "exists: $p"
    else
      warn "missing: $p"
    fi
  done
  mkdir -p "$SCRIPT_DIR/data/logbook" "$SCRIPT_DIR/data/control" 2>/dev/null || true
  if [[ -w "$SCRIPT_DIR/data/logbook" && -w "$SCRIPT_DIR/data/control" ]]; then
    notice "data dirs writable"
  else
    warn "data dirs not writable"
  fi

  # Compose services
  local svcs
  svcs="$(list_services || true)"
  if [[ -n "$svcs" ]]; then
    notice "services: $(echo "$svcs" | tr '\n' ' ')"
  else
    warn "no services detected in $COMPOSE_FILE"
  fi
  notice "Diagnostics complete."
}

run_dev() {
  local what="$1"; shift || true
  case "$what" in
    bot)
      notice "Starting local bot (python -m src.supervisor)"
      PYTHONPATH="$SCRIPT_DIR" run_cmd python3 -m src.supervisor "$@"
      ;;
    ui)
      local port="8501"
      while (($#)); do
        case "$1" in
          --port) shift; port="$1" ;;
          *) break ;;
        esac
        shift || true
      done
      notice "Starting local UI on :$port"
      PYTHONPATH="$SCRIPT_DIR" run_cmd streamlit run "$SCRIPT_DIR/ui/ui_app.py" --server.port="$port" --server.address=0.0.0.0 "$@"
      ;;
    *)
      die "Unknown dev target: ${what:-<empty>}"
      ;;
  esac
}

run_command() {
  local cmd="$1"; shift || true
  case "$cmd" in
    up)
      local mode="detached"
      if [[ "${1:-}" == "fg" || "${1:-}" == "foreground" ]]; then
        mode="fg"; shift || true
      fi
      if [[ "$mode" == "fg" ]]; then action_up_fg; else action_up_detached; fi
      ;;
    down) action_down ;;
    ps) action_ps ;;
    build)
      if [[ "${1:-}" == "--no-cache" ]]; then action_rebuild_nocache; else action_build; fi
      ;;
    pull) action_pull ;;
    logs)
      local s="${1:-}"
      if [[ -n "$s" ]]; then shift || true; else s="$(choose_service)"; fi
      if [[ "${1:-}" == "follow" ]]; then dc logs -f "$s"; else dc logs --tail=200 "$s"; fi
      ;;
    restart)
      local s="${1:-}"; if [[ -z "$s" ]]; then s="$(choose_service)"; fi; dc restart "$s"
      ;;
    stop)
      local s="${1:-}"; if [[ -z "$s" ]]; then s="$(choose_service)"; fi; dc stop "$s"
      ;;
    start-all)
      action_start_all
      ;;
    stop-all)
      action_stop_all
      ;;
    restart-all)
      action_restart_all
      ;;
    build-restart)
      local target="${1:-}"
      if [[ -z "$target" ]]; then
        target="$(choose_service)"
      fi
      if [[ "$target" == "all" ]]; then
        dc up -d --build
      else
        dc up -d --build "$target"
      fi
      ;;
    exec)
      local s="${1:-}"; if [[ -z "$s" ]]; then s="$(choose_service)"; else shift || true; fi
      if [[ "${1:-}" == "bash" || "${1:-}" == "sh" || "${1:-}" == "python" ]]; then
        local shell="$1"; shift || true
        if [[ "$shell" == "bash" || "$shell" == "sh" ]]; then dc exec -it "$s" "$shell"; else dc exec "$s" sh -lc "$shell"; fi
      else
        if [[ "${1:-}" == "--" ]]; then shift || true; fi
        if (($#)); then dc exec "$s" sh -lc "$*"; else dc exec -it "$s" sh; fi
      fi
      ;;
    config) action_config ;;
    prune-docker) action_prune ;;
    doctor) doctor ;;
    dev) run_dev "$@" ;;
    *)
      die "Unknown command: $cmd"
      ;;
  esac
}

main() {
  parse_args "$@"
  if [[ ! -f "$COMPOSE_FILE" ]]; then
    die "Compose file not found: $COMPOSE_FILE"
  fi
  trap 'printf "\n"; warn "Interrupted."; exit 130' INT
  DC_BASE="$(detect_dc)" || true
  if [[ -n "$CMD" ]]; then
    # Safely expand CMD_ARGS under 'set -u' when it may be unset or empty
    if [[ ${#CMD_ARGS[@]:-0} -gt 0 ]]; then
      run_command "$CMD" "${CMD_ARGS[@]}"
    else
      run_command "$CMD"
    fi
  else
    interactive_menu
  fi
}

main "$@"


