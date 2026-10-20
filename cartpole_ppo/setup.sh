#!/usr/bin/env bash
# 只在本项目内创建工具、环境和缓存，不运行 shell init，不使用系统 pip。
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
export MAMBA_ROOT_PREFIX="$PROJECT_DIR/.mamba"
export CONDA_PKGS_DIRS="$PROJECT_DIR/.mamba/pkgs"
export CONDA_ENVS_PATH="$PROJECT_DIR/.mamba/envs"
export XDG_CACHE_HOME="$PROJECT_DIR/.cache"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"
export PIP_CONFIG_FILE=/dev/null
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
mkdir -p .tools .cache

if [[ ! -x .tools/bin/micromamba ]]; then
    # 本安装脚本对应当前机器的 Linux x86_64 平台。
    curl -fL --retry 3 https://micro.mamba.pm/api/micromamba/linux-64/2.9.0 \
        -o .tools/micromamba.tar.bz2
    tar -xjf .tools/micromamba.tar.bz2 -C .tools bin/micromamba
fi

if [[ ! -x .env/bin/python ]]; then
    # 使用机器已有的 bwrap，把 micromamba 的用户环境登记表放进临时目录。
    # 其余系统目录只读，只有本项目可写，避免改动 ~/.conda/environments.txt。
    bwrap --ro-bind / / --dev /dev --bind "$PROJECT_DIR" "$PROJECT_DIR" \
        --tmpfs /tmp --tmpfs "$HOME/.conda" \
        .tools/bin/micromamba --no-rc create -y -p "$PROJECT_DIR/.env" \
        --override-channels -c conda-forge python=3.11 pip
fi

# -I 使用隔离模式，所有包都装进 micromamba 创建的 .env。
.env/bin/python -I -m pip --isolated install \
    --cache-dir "$PIP_CACHE_DIR" --index-url https://download.pytorch.org/whl/cpu \
    'torch==2.8.0+cpu'
.env/bin/python -I -m pip --isolated install \
    --cache-dir "$PIP_CACHE_DIR" --index-url https://pypi.org/simple \
    -r requirements.txt
.env/bin/python -I -m pip --isolated check
# list 输出包名和版本，避免 freeze 把 Conda 构建机上的本地路径写进锁定清单。
.env/bin/python -I -m pip --isolated list --format=freeze --disable-pip-version-check > requirements-lock.txt
echo '环境准备完成。运行：bash run.sh all'
