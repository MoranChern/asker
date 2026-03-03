#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# One-click environment setup (no args).
# Creates/updates conda env: asker
# - Python: 3.13
# - CUDA toolkit (conda): 12.4 (best-effort)
# - Installs: neo4j driver, llama-cpp-python (CUDA preferred)
#
# Notes:
# - Assumes you already have conda installed and available on PATH.
# - Uses China-friendly mirrors (TUNA) for conda-forge/nvidia + pip.
# ============================================================

ENV_NAME="asker"
PY_VER="3.13"
CUDA_VER="12.4"

# China-friendly mirrors (TUNA)
CONDA_FORGE_CH="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"
NVIDIA_CH="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/nvidia"
PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "[*] Target conda env: ${ENV_NAME}"
echo "[*] Python version:   ${PY_VER}"
echo "[*] CUDA toolkit:     ${CUDA_VER} (best-effort)"
echo

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found on PATH."
  echo "        Please install Miniconda/Anaconda/Miniforge first, then re-run."
  exit 1
fi

# Ensure we can use 'conda activate' in a non-interactive script
eval "$(conda shell.bash hook)"

# Create env if missing
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[+] Conda env '${ENV_NAME}' already exists. Will update packages."
else
  echo "[+] Creating conda env '${ENV_NAME}'..."
  conda create -y -n "${ENV_NAME}" -c "${CONDA_FORGE_CH}" "python=${PY_VER}" pip
fi

echo "[+] Installing build/runtime dependencies via conda..."
conda install -y -n "${ENV_NAME}" -c "${CONDA_FORGE_CH}" \
  cmake ninja make pkg-config git \
  "cxx-compiler"

echo "[+] Installing CUDA toolkit via conda (best-effort)..."
set +e
conda install -y -n "${ENV_NAME}" -c "${NVIDIA_CH}" -c "${CONDA_FORGE_CH}" "cuda-toolkit=${CUDA_VER}"
CUDA_OK=$?
set -e
if [[ ${CUDA_OK} -ne 0 ]]; then
  echo "[!] cuda-toolkit=${CUDA_VER} install failed (channel availability / platform mismatch / already on system)."
  echo "    Continuing without conda CUDA toolkit."
fi

echo "[+] Upgrading pip toolchain..."
conda run -n "${ENV_NAME}" python -m pip config set global.index-url "${PIP_INDEX}" >/dev/null 2>&1 || true
conda run -n "${ENV_NAME}" python -m pip install -U pip setuptools wheel scikit-build-core

echo "[+] Installing Python deps..."
conda run -n "${ENV_NAME}" python -m pip install -U neo4j

echo "[+] Installing llama-cpp-python (CUDA preferred)..."
set +e
conda run -n "${ENV_NAME}" env CMAKE_ARGS="-DGGML_CUDA=ON" FORCE_CMAKE=1 \
  python -m pip install --no-cache-dir -U llama-cpp-python
LLAMA_OK=$?
set -e
if [[ ${LLAMA_OK} -ne 0 ]]; then
  echo "[!] CUDA build failed, falling back to CPU build..."
  conda run -n "${ENV_NAME}" python -m pip install --no-cache-dir -U llama-cpp-python
fi

echo "[+] Quick import test..."
conda run -n "${ENV_NAME}" python - <<'PY'
import sys
import neo4j
import llama_cpp
print("OK:", sys.version.split()[0])
print("neo4j:", neo4j.__version__)
print("llama_cpp:", getattr(llama_cpp, "__version__", "unknown"))
PY

echo
echo "[✓] Environment ready."
echo "    Activate:  conda activate ${ENV_NAME}"
echo "    Run demo:  python neo4j_rag_app.py"

echo "Now testing if cuda version is available in Python (llama-cpp-python)..."
conda activate asker
python - <<'PY'
from llama_cpp import llama_cpp
print("GGML_CUDA:", bool(getattr(llama_cpp, "ggml_cuda_init", None)))
PY
echo "
GGML_CUDA: True means CUDA is available and will be used for llama-cpp-python. False means CPU-only."