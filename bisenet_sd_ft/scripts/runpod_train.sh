#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/runpod_train.sh <config-path> [auto_action_hours] [none|terminate_on_success|terminate_on_exit|stop_on_success|stop_on_exit]"
  exit 1
fi

CONFIG_PATH="$1"
AUTO_ACTION_HOURS="${2:-}"
TERMINATE_MODE="${3:-none}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="outputs/launcher_logs"
LOG_PATH="${LOG_DIR}/train_${TIMESTAMP}.log"
RUNNER_PATH="${LOG_DIR}/runner_${TIMESTAMP}.sh"

mkdir -p "${LOG_DIR}"
export PYTHONUNBUFFERED=1

cat > "${RUNNER_PATH}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
python src/train.py --config "${CONFIG_PATH}" >> "${LOG_PATH}" 2>&1
EXIT_CODE=\$?
if [[ "${TERMINATE_MODE}" == "terminate_on_success" && "\${EXIT_CODE}" -eq 0 ]]; then
  if [[ -n "\${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null 2>&1; then
    runpodctl remove pod "\${RUNPOD_POD_ID}" >> "${LOG_PATH}" 2>&1 || true
  fi
fi
if [[ "${TERMINATE_MODE}" == "terminate_on_exit" ]]; then
  if [[ -n "\${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null 2>&1; then
    runpodctl remove pod "\${RUNPOD_POD_ID}" >> "${LOG_PATH}" 2>&1 || true
  fi
fi
if [[ "${TERMINATE_MODE}" == "stop_on_success" && "\${EXIT_CODE}" -eq 0 ]]; then
  if [[ -n "\${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null 2>&1; then
    runpodctl stop pod "\${RUNPOD_POD_ID}" >> "${LOG_PATH}" 2>&1 || true
  fi
fi
if [[ "${TERMINATE_MODE}" == "stop_on_exit" ]]; then
  if [[ -n "\${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null 2>&1; then
    runpodctl stop pod "\${RUNPOD_POD_ID}" >> "${LOG_PATH}" 2>&1 || true
  fi
fi
exit "\${EXIT_CODE}"
EOF

chmod +x "${RUNNER_PATH}"
nohup bash "${RUNNER_PATH}" > /dev/null 2>&1 &
TRAIN_PID=$!

echo "Training started"
echo "PID: ${TRAIN_PID}"
echo "Log: ${LOG_PATH}"
echo "Terminate mode: ${TERMINATE_MODE}"

if [[ -n "${AUTO_ACTION_HOURS}" ]]; then
  if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
    echo "RUNPOD_POD_ID is not set. Skipping auto action scheduling."
    exit 0
  fi
  if ! command -v runpodctl >/dev/null 2>&1; then
    echo "runpodctl not found. Skipping auto action scheduling."
    exit 0
  fi
  if [[ "${TERMINATE_MODE}" == stop_* ]]; then
    nohup bash -lc "sleep ${AUTO_ACTION_HOURS}h; runpodctl stop pod ${RUNPOD_POD_ID}" > "${LOG_DIR}/action_${TIMESTAMP}.log" 2>&1 &
    echo "Auto-stop scheduled after ${AUTO_ACTION_HOURS} hour(s)."
  else
    nohup bash -lc "sleep ${AUTO_ACTION_HOURS}h; runpodctl remove pod ${RUNPOD_POD_ID}" > "${LOG_DIR}/action_${TIMESTAMP}.log" 2>&1 &
    echo "Auto-terminate scheduled after ${AUTO_ACTION_HOURS} hour(s)."
  fi
fi
