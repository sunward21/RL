#!/usr/bin/env bash
# 显式调用项目环境里的 Python；无需激活，也不改变当前终端的环境。
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$PROJECT_DIR/.env/bin/python" ]]; then
    echo '请先运行 bash setup.sh 创建独立环境。' >&2
    exit 1
fi
export XDG_CACHE_HOME="$PROJECT_DIR/.cache"
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
export PYGAME_HIDE_SUPPORT_PROMPT=1
exec "$PROJECT_DIR/.env/bin/python" -I "$PROJECT_DIR/cartpole.py" "$@"
