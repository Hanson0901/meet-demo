#!/usr/bin/env bash

set -Eeuo pipefail

SERVICE_NAME="${MEETING_SERVICE_NAME:-meeting-assistant.service}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${MEETING_PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${MEETING_PYTHON_BIN:-/home/cgu-csie/miniconda3/envs/py3_10/bin/python}"
MODEL_PATH="${LLAMA_CPP_MODEL:-/home/cgu-csie/qwen3-4b-instruct-2507-q8_0.gguf}"
ENV_FILE="${MEETING_ENV_FILE:-${XDG_CONFIG_HOME:-/home/cgu-csie/.config}/meeting-assistant.env}"
HEALTH_URL="${MEETING_HEALTH_URL:-http://127.0.0.1:8080/health}"

usage() {
    cat <<'EOF'
用法：
  scripts/meeting-assistant-service.sh start    開啟服務
  scripts/meeting-assistant-service.sh stop     關閉服務
  scripts/meeting-assistant-service.sh restart  重啟服務
  scripts/meeting-assistant-service.sh status   顯示服務狀態
  scripts/meeting-assistant-service.sh logs [N] 查看最近 N 筆 log 並持續追蹤
  scripts/meeting-assistant-service.sh health   檢查 Web 健康狀態

可用環境變數：
  MEETING_SERVICE_NAME  systemd unit 名稱
  MEETING_PROJECT_DIR   專案目錄
  MEETING_PYTHON_BIN    Python 執行檔
  LLAMA_CPP_MODEL       GGUF 模型路徑
  MEETING_ENV_FILE      systemd EnvironmentFile 路徑
  MEETING_HEALTH_URL    健康檢查 URL
EOF
}

prepare_user_bus() {
    if [[ -n "${XDG_RUNTIME_DIR:-}" ]]; then
        return
    fi

    local runtime_dir="/run/user/$(id -u)"
    if [[ -d "$runtime_dir" ]]; then
        export XDG_RUNTIME_DIR="$runtime_dir"
    fi
}

systemctl_user() {
    prepare_user_bus
    systemctl --user "$@"
}

unit_exists() {
    local load_state
    load_state="$(systemctl_user show "$SERVICE_NAME" -p LoadState --value 2>/dev/null || true)"
    [[ -n "$load_state" && "$load_state" != "not-found" ]]
}

unit_is_transient() {
    [[ "$(systemctl_user show "$SERVICE_NAME" -p Transient --value 2>/dev/null || true)" == "yes" ]]
}

start_transient_unit() {
    if [[ ! -x "$PYTHON_BIN" ]]; then
        printf '找不到 Python 執行檔：%s\n' "$PYTHON_BIN" >&2
        printf '請設定 MEETING_PYTHON_BIN，或先建立永久 systemd unit。\n' >&2
        return 1
    fi

    local -a environment_args=(
        "--setenv=LLAMA_CPP_CONTEXT=${LLAMA_CPP_CONTEXT:-16384}"
        "--setenv=PYTHONUNBUFFERED=1"
    )
    if [[ -f "$MODEL_PATH" ]]; then
        environment_args+=("--setenv=LLAMA_CPP_MODEL=${MODEL_PATH}")
    else
        printf '警告：找不到 GGUF 模型：%s\n' "$MODEL_PATH" >&2
    fi

    local -a env_file_args=()
    if [[ -f "$ENV_FILE" ]]; then
        env_file_args+=("--property=EnvironmentFile=${ENV_FILE}")
    else
        # systemd does not source ~/.bashrc.  When this helper is launched
        # from an interactive shell, inherit exported settings without putting
        # secret values on the systemd-run command line.
        local name
        for name in CGULLM_API_KEY CGULLM_BASE_URL CGULLM_MODEL CGULLM_TIMEOUT MEETING_ADMIN_TOKEN; do
            if [[ -n "${!name:-}" ]]; then
                environment_args+=("--setenv=${name}")
            fi
        done
    fi

    printf '建立一次性 user systemd unit：%s\n' "$SERVICE_NAME"
    systemctl_user stop "$SERVICE_NAME" >/dev/null 2>&1 || true
    prepare_user_bus
    systemd-run --user \
        --unit="$SERVICE_NAME" \
        --property=Restart=on-failure \
        --property=RestartSec=5s \
        --working-directory="$PROJECT_DIR" \
        "${env_file_args[@]}" \
        "${environment_args[@]}" \
        "$PYTHON_BIN" -u -m meeting_assistant.cli web --host 0.0.0.0 --port 8080
}

start_service() {
    if unit_exists; then
        systemctl_user start "$SERVICE_NAME"
    else
        start_transient_unit
    fi
    printf '服務已啟動：%s\n' "$SERVICE_NAME"
}

stop_service() {
    systemctl_user stop "$SERVICE_NAME"
    printf '服務已停止：%s\n' "$SERVICE_NAME"
}

restart_service() {
    if unit_exists && unit_is_transient; then
        # A transient unit keeps the environment captured at creation time.
        # Recreate it so a changed ~/.bashrc or EnvironmentFile is applied.
        start_transient_unit
    elif unit_exists; then
        systemctl_user restart "$SERVICE_NAME"
    else
        start_transient_unit
    fi
    printf '服務已重啟：%s\n' "$SERVICE_NAME"
}

show_logs() {
    local lines="${1:-100}"
    if [[ ! "$lines" =~ ^[0-9]+$ ]]; then
        printf 'log 筆數必須是非負整數：%s\n' "$lines" >&2
        return 64
    fi
    prepare_user_bus
    journalctl --user -u "$SERVICE_NAME" -n "$lines" -f -o cat
}

main() {
    local action="${1:-status}"
    case "$action" in
        start|open)
            start_service
            ;;
        stop|close)
            stop_service
            ;;
        restart)
            restart_service
            ;;
        status)
            systemctl_user status "$SERVICE_NAME" --no-pager
            ;;
        logs)
            show_logs "${2:-100}"
            ;;
        health)
            curl --fail --silent --show-error "$HEALTH_URL"
            printf '\n'
            ;;
        help|-h|--help)
            usage
            ;;
        *)
            printf '未知操作：%s\n\n' "$action" >&2
            usage >&2
            return 64
            ;;
    esac
}

main "$@"
