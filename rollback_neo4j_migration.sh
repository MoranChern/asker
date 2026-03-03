#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Roll back the last Neo4j migration done by migrate script.
# - Restores docker-compose.yml from backup
# - Stops/starts ONLY kg-neo4j
# - Does NOT delete the migrated data dir or dump backups
# ============================================================

COMPOSE_FILE="docker-compose.yml"
SERVICE_NAME="kg-neo4j"
STATE_FILE="./neo4j/migration_backups/latest_state.env"

log() { echo -e "[rollback] $*"; }

ensure_in_project_root() {
  if [[ -f "${COMPOSE_FILE}" ]]; then
    return 0
  fi
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "${script_dir}/${COMPOSE_FILE}" ]]; then
    cd "${script_dir}"
    return 0
  fi
  echo "[ERROR] Cannot find ${COMPOSE_FILE} in current dir or script dir."
  exit 1
}

main() {
  ensure_in_project_root

  if ! docker compose version >/dev/null 2>&1; then
    echo "[ERROR] 'docker compose' not available. Please install Docker Compose v2."
    exit 1
  fi

  if [[ ! -f "${STATE_FILE}" ]]; then
    echo "[ERROR] State file not found: ${STATE_FILE}"
    echo "        Nothing to rollback (or you ran migration script in another directory)."
    exit 1
  fi

  # shellcheck disable=SC1090
  source "${STATE_FILE}"

  if [[ -z "${COMPOSE_BACKUP:-}" || ! -f "${COMPOSE_BACKUP}" ]]; then
    echo "[ERROR] COMPOSE_BACKUP missing or not found in state: ${COMPOSE_BACKUP:-<empty>}"
    exit 1
  fi

  log "Stopping only ${SERVICE_NAME}..."
  docker compose stop "${SERVICE_NAME}" || true

  log "Restoring ${COMPOSE_FILE} from backup: ${COMPOSE_BACKUP}"
  cp -f "${COMPOSE_BACKUP}" "${COMPOSE_FILE}"

  log "Starting only ${SERVICE_NAME} with restored compose..."
  docker compose up -d "${SERVICE_NAME}"

  echo
  log "DONE."
  log "Notes:"
  log "  - Old data dir (should be used again): ${OLD_DATA_DIR:-./neo4j/data}"
  log "  - Migrated data dir kept (not deleted): ${NEW_DATA_DIR:-./neo4j/data_2026}"
  log "  - Dump backup kept: ${DUMP_FILE:-<unknown>}"
}

main
