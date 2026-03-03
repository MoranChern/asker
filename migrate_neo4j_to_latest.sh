#!/usr/bin/env bash
set -euo pipefail

# ===================== Config =====================
DB_NAME="neo4j"

OLD_IMAGE="neo4j:4.4.11-community"
HOP_IMAGE="neo4j:5.26.21-community"
NEW_IMAGE="neo4j:2026.01.4-community"

COMPOSE_FILE="docker-compose.yml"
SERVICE_NAME="kg-neo4j"

OLD_DATA_DIR="./neo4j/data"
NEW_DATA_DIR="./neo4j/data_2026"

BACKUP_ROOT="./neo4j/migration_backups"

# ===================== Helpers =====================
log() { echo -e "[migrate] $*"; }
die() { echo -e "[ERROR] $*" >&2; exit 1; }

require_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"; }

compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose "$@"
  else
    die "Neither 'docker compose' nor 'docker-compose' is available."
  fi
}

ensure_in_project_root() {
  if [[ -f "${COMPOSE_FILE}" ]]; then return 0; fi
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "${script_dir}/${COMPOSE_FILE}" ]]; then
    cd "${script_dir}"
    return 0
  fi
  die "Cannot find ${COMPOSE_FILE} in current dir or script dir."
}

ensure_writable_dir() {
  local d="$1"
  mkdir -p "$d"
  chmod -R a+rwx "$d" >/dev/null 2>&1 || true
}

ensure_image_present() {
  local img="$1"
  if ! docker image inspect "$img" >/dev/null 2>&1; then
    log "Image not found locally: $img  (will try to pull)"
    docker pull "$img" || die "Failed to pull $img"
  fi
}

detect_data_uidgid() {
  local probe=""
  if [[ -e "${OLD_DATA_DIR}/databases/${DB_NAME}" ]]; then
    probe="${OLD_DATA_DIR}/databases/${DB_NAME}"
  elif [[ -e "${OLD_DATA_DIR}/databases/system" ]]; then
    probe="${OLD_DATA_DIR}/databases/system"
  else
    probe="$(find "${OLD_DATA_DIR}" -maxdepth 3 -type f 2>/dev/null | head -n 1 || true)"
  fi
  if [[ -z "$probe" ]]; then
    stat -c '%u:%g' "${OLD_DATA_DIR}" 2>/dev/null || echo ""
    return 0
  fi
  stat -c '%u:%g' "$probe" 2>/dev/null || echo ""
}

# 用指定 UID:GID + 指定源目录来 dump（源目录以 RW 挂载：避免 neo4j-admin 需要写 lock 时失败）
try_dump_from_dir() {
  local uidgid="$1"
  local src_dir="$2"
  local dump_file="$3"
  local backups_dir="$4"

  log "Trying dump as user ${uidgid} from ${src_dir} (RW mount) ..."
  set +e
  docker run --rm -i \
    -u "${uidgid}" \
    --entrypoint=neo4j-admin \
    -v "$(pwd)/${src_dir#./}:/data" \
    -v "$(pwd)/${backups_dir#./}:/backups" \
    "${OLD_IMAGE}" \
    dump --database="${DB_NAME}" --to="/backups/${DB_NAME}.dump"
  local rc=$?
  set -e

  [[ $rc -eq 0 && -f "${dump_file}" ]]
}

# 把 OLD_DATA_DIR 复制到本地临时目录（在容器里用 tar 读 /src 写 /dst）
snapshot_to_local_tmp() {
  local uidgid="$1"
  local snapshot_dir="$2"

  ensure_writable_dir "$snapshot_dir"

  log "Copying ${OLD_DATA_DIR} -> ${snapshot_dir} (as ${uidgid}) ..."
  docker run --rm -i \
    -u "${uidgid}" \
    --entrypoint=sh \
    -v "$(pwd)/${OLD_DATA_DIR#./}:/src:ro" \
    -v "${snapshot_dir}:/dst" \
    "${OLD_IMAGE}" \
    -c 'cd /src && tar cf - . | (cd /dst && tar xpf - --no-same-owner)'
}

patch_compose_inplace() {
  local compose_path="$1"
  local backup_path="$2"
  local new_image="$3"
  local old_data="$4"
  local new_data="$5"

  cp -f "$compose_path" "$backup_path"
  log "Backed up ${compose_path} -> ${backup_path}"

  python3 - "$compose_path" "$new_image" "$old_data" "$new_data" <<'PY'
import sys, re
compose_path = sys.argv[1]
new_image    = sys.argv[2]
old_data     = sys.argv[3]
new_data     = sys.argv[4]

txt = open(compose_path, "r", encoding="utf-8").read()
m = re.search(r'(?ms)^(\s*)kg-neo4j:\s*\n(.*?)(?=^\1\S|\Z)', txt)
if not m:
    raise SystemExit("Could not find service block: kg-neo4j")

indent = m.group(1)
block  = m.group(0)

img_line_re = re.compile(r'(?m)^' + re.escape(indent) + r'\s+image:\s*.*$')
if img_line_re.search(block):
    block = img_line_re.sub(indent + "  image: " + new_image, block, count=1)
else:
    block = block.replace(indent + "kg-neo4j:\n",
                          indent + "kg-neo4j:\n" + indent + "  image: " + new_image + "\n",
                          1)

block = re.sub(re.escape(old_data) + r'(?=:/data\b)', new_data, block)

txt2 = txt[:m.start()] + block + txt[m.end():]
open(compose_path, "w", encoding="utf-8").write(txt2)
print("Compose patched: kg-neo4j image + /data volume updated.")
PY
}

main() {
  ensure_in_project_root
  require_cmd docker
  require_cmd python3

  [[ -d "${OLD_DATA_DIR}" ]] || die "Old data directory not found: ${OLD_DATA_DIR}"

  local ts run_dir backups_dir dump_file compose_bak
  ts="$(date +%Y%m%d_%H%M%S)"
  run_dir="${BACKUP_ROOT}/${ts}"
  backups_dir="${run_dir}/backups"
  dump_file="${backups_dir}/${DB_NAME}.dump"
  compose_bak="${run_dir}/docker-compose.yml.bak"

  log "This will migrate Neo4j data:"
  log "  ${OLD_IMAGE} -> ${HOP_IMAGE} -> ${NEW_IMAGE}"
  log "Old data dir kept as-is: ${OLD_DATA_DIR}"
  log "New data dir will be created: ${NEW_DATA_DIR}"
  log "Run dir: ${run_dir}"
  echo

  # New dir must be empty (or not exist)
  if [[ -d "${NEW_DATA_DIR}" ]] && [[ -n "$(ls -A "${NEW_DATA_DIR}" 2>/dev/null || true)" ]]; then
    die "New data directory exists and is not empty: ${NEW_DATA_DIR}
To re-run: move/delete it (it's the *new* dir), then run again."
  fi

  ensure_writable_dir "${BACKUP_ROOT}"
  ensure_writable_dir "${backups_dir}"
  ensure_writable_dir "${NEW_DATA_DIR}"

  ensure_image_present "${OLD_IMAGE}"
  ensure_image_present "${HOP_IMAGE}"
  ensure_image_present "${NEW_IMAGE}"

  log "Stopping only ${SERVICE_NAME} (other services untouched)..."
  compose stop "${SERVICE_NAME}" || true

  # ---- dump user candidates ----
  local detected_uidgid
  detected_uidgid="$(detect_data_uidgid || true)"

  # de-dup candidates
  declare -A seen
  local candidates=()
  for u in "${detected_uidgid:-}" "7474:7474" "0:0"; do
    [[ -z "$u" ]] && continue
    if [[ -z "${seen[$u]+x}" ]]; then
      candidates+=("$u")
      seen[$u]=1
    fi
  done

  # ---- 1) Try dump directly from original dir (RW mount) ----
  log "Creating offline dump -> ${dump_file}"
  local ok=0
  for uidgid in "${candidates[@]}"; do
    if try_dump_from_dir "$uidgid" "${OLD_DATA_DIR}" "${dump_file}" "${backups_dir}"; then
      ok=1
      log "Dump OK (direct, user ${uidgid})."
      break
    fi
  done

  # ---- 1b) If still fails, snapshot to local tmp and dump from snapshot ----
  local snapshot_dir=""
  if [[ $ok -ne 1 ]]; then
    snapshot_dir="${TMPDIR:-/tmp}/${USER}/neo4j_migrate_${ts}/data_snapshot"
    log "Direct dump failed. Will snapshot to local tmp and retry dump from snapshot:"
    log "  ${snapshot_dir}"

    # Use best non-root first for snapshot copy; if that fails, try root
    local copied=0
    for uidgid in "${candidates[@]}"; do
      set +e
      snapshot_to_local_tmp "$uidgid" "$snapshot_dir"
      rc=$?
      set -e
      if [[ $rc -eq 0 ]]; then
        copied=1
        log "Snapshot copy OK (user ${uidgid})."
        break
      fi
    done
    [[ $copied -eq 1 ]] || die "Snapshot copy failed too. This usually means the host filesystem denies container reads (NFS/ACL/rootless mapping). You need to move the project to a local disk path (e.g. /tmp or local scratch) and run there."

    # Now dump from snapshot (RW mount)
    for uidgid in "${candidates[@]}"; do
      if try_dump_from_dir "$uidgid" "${snapshot_dir}" "${dump_file}" "${backups_dir}"; then
        ok=1
        log "Dump OK (from snapshot, user ${uidgid})."
        break
      fi
    done
  fi

  [[ $ok -eq 1 ]] || die "Dump failed even after snapshot."

  # ---- 2) Load into Neo4j 5 data dir ----
  log "Loading dump into new data dir using ${HOP_IMAGE}"
  local host_uid host_gid
  host_uid="$(id -u)"
  host_gid="$(id -g)"

  docker run --rm -i \
    -u "${host_uid}:${host_gid}" \
    --entrypoint=neo4j-admin \
    -v "$(pwd)/${NEW_DATA_DIR#./}:/data" \
    -v "$(pwd)/${backups_dir#./}:/backups:ro" \
    "${HOP_IMAGE}" \
    database load "${DB_NAME}" --from-path="/backups" --overwrite-destination=true

  # ---- 3) Migrate store to Neo4j 5 format ----
  log "Migrating store to Neo4j 5 format..."
  set +e
  docker run --rm -i \
    -u "${host_uid}:${host_gid}" \
    --entrypoint=neo4j-admin \
    -v "$(pwd)/${NEW_DATA_DIR#./}:/data" \
    "${HOP_IMAGE}" \
    database migrate "${DB_NAME}"
  mig_rc=$?
  set -e

  if [[ $mig_rc -ne 0 ]]; then
    log "First migrate failed; retrying with --force-btree-indexes-to-range ..."
    docker run --rm -i \
      -u "${host_uid}:${host_gid}" \
      --entrypoint=neo4j-admin \
      -v "$(pwd)/${NEW_DATA_DIR#./}:/data" \
      "${HOP_IMAGE}" \
      database migrate "${DB_NAME}" --force-btree-indexes-to-range
  fi

  chmod -R a+rwx "${NEW_DATA_DIR}" >/dev/null 2>&1 || true

  # ---- 4) Patch compose + start latest ----
  log "Patching ${COMPOSE_FILE}: set image=${NEW_IMAGE}, volume ${NEW_DATA_DIR}:/data for ${SERVICE_NAME}"
  patch_compose_inplace "${COMPOSE_FILE}" "${compose_bak}" "${NEW_IMAGE}" "${OLD_DATA_DIR}" "${NEW_DATA_DIR}"

  log "Starting only ${SERVICE_NAME}..."
  compose up -d "${SERVICE_NAME}"

  echo
  log "DONE."
  log "Dump: ${dump_file}"
  log "Compose backup: ${compose_bak}"
  log "New data dir: ${NEW_DATA_DIR}"
  if [[ -n "${snapshot_dir}" ]]; then
    log "Snapshot dir (if created): ${snapshot_dir}"
  fi
}

main "$@"
