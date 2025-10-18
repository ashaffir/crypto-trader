#!/usr/bin/env bash

# Interactive Docker Compose manager for this project.
# - Minimal dependencies; prefers `docker compose`, falls back to `docker-compose`.
# - Interactive menu for build, up, down, restart, logs, exec, status, etc.
# - Supports non-interactive flags: --help, --dry-run, --self-test, --file.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
COMPOSE_FILE="${DEFAULT_COMPOSE_FILE}"
DRY_RUN="false"
FORCE_DOCKER_TOOL=""  # internal/testing override: docker|docker-compose

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
  # Determine docker compose CLI
  local dc_cmd=""
  if [[ -n "$FORCE_DOCKER_TOOL" ]]; then
    case "$FORCE_DOCKER_TOOL" in
      docker)         dc_cmd="docker compose" ;;
      docker-compose) dc_cmd="docker-compose" ;;
      *) die "Unknown FORCE_DOCKER_TOOL=$FORCE_DOCKER_TOOL" ;;
    esac
  elif have docker && docker compose version >/dev/null 2>&1; then
    dc_cmd="docker compose"
  elif have docker-compose && docker-compose version >/dev/null 2>&1; then
    dc_cmd="docker-compose"
  else
    dc_cmd=""  # not found
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
    if services_out=$(dc config --services 2>/dev/null); then
      printf "%s\n" "$services_out"
      return 0
    fi
  fi
  # Fallback: naive YAML parse of service names under top-level 'services:'
  # Note: this is a simple heuristic; adequate for local self-test without docker.
  awk '
    $0 ~ /^services:/ {in_services=1; next}
    in_services && match($0, /^[[:space:]]{2}([A-Za-z0-9_.-]+):/, m) {print m[1]}
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
  local services=("$(list_services | tr '\n' ' ')")
  # word-split into array
  # shellcheck disable=SC2206
  services=(${services})
  if ((${#services[@]}==0)); then
    die "No services found in compose file: $COMPOSE_FILE"
  fi
  notice "Select a service:"
  select svc in "${services[@]}"; do
    if [[ -n "${svc:-}" ]]; then
      printf "%s" "$svc"
      return 0
    fi
    warn "Invalid selection."
  done
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
 15) Exit
EOF
}

interactive_menu() {
  local choice
  while true; do
    print_menu
    read -r -p "Enter choice [1-15]: " choice || true
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
      15) notice "Bye!"; break ;;
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
  -h, --help          Show this help

Without options, an interactive menu is shown.
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
      -h|--help)
        usage; exit 0
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
    shift || true
  done
}

main() {
  parse_args "$@"
  if [[ ! -f "$COMPOSE_FILE" ]]; then
    die "Compose file not found: $COMPOSE_FILE"
  fi
  trap 'printf "\n"; warn "Interrupted."; exit 130' INT
  DC_BASE="$(detect_dc)" || true
  interactive_menu
}

main "$@"


