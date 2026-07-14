#!/usr/bin/env bash
set -u

PROJECT_ROOT="."
OPENPCDET_ROOT="$PROJECT_ROOT/external/OpenPCDet"
PYTHON_BIN="${PYTHON_BIN:-python}"

CUDA_HOME="/usr/local/cuda"
export CUDA_HOME
export CUDA_PATH="$CUDA_HOME"
export NUMBAPRO_NVVM="$CUDA_HOME/nvvm/lib64/libnvvm.so"
export NUMBAPRO_LIBDEVICE="$CUDA_HOME/nvvm/libdevice/libdevice.10.bc"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
export NUMBA_FORCE_CUDA_CC="8.9"
export LD_LIBRARY_PATH="$CUDA_HOME/nvvm/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"

CFG_FILE=""
CKPT=""
EXTRA_TAG="shell_retry"
EVAL_TAG="shell_retry"
BATCH_SIZE="2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cfg-file)
      CFG_FILE="$2"
      shift 2
      ;;
    --ckpt)
      CKPT="$2"
      shift 2
      ;;
    --extra-tag)
      EXTRA_TAG="$2"
      shift 2
      ;;
    --eval-tag)
      EVAL_TAG="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

echo "=== shell-level CUDA/NVVM environment ==="
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "OPENPCDET_ROOT=$OPENPCDET_ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_PATH=$CUDA_PATH"
echo "NUMBAPRO_NVVM=$NUMBAPRO_NVVM"
echo "NUMBAPRO_LIBDEVICE=$NUMBAPRO_LIBDEVICE"
echo "XLA_FLAGS=$XLA_FLAGS"
echo "NUMBA_FORCE_CUDA_CC=$NUMBA_FORCE_CUDA_CC"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo "=== python executable check ==="
which python || true
"$PYTHON_BIN" - <<'PY'
import sys
print("sys.executable=", sys.executable)
PY

echo "=== ldd libnvvm ==="
ldd "$NUMBAPRO_NVVM" || exit 11

echo "=== pip package check ==="
"$PYTHON_BIN" -m pip show nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 numba llvmlite || true

echo "=== sanity check ==="
"$PYTHON_BIN" - <<'PY'
import os
import ctypes
print("CUDA_HOME=", os.environ.get("CUDA_HOME"))
print("NUMBAPRO_NVVM=", os.environ.get("NUMBAPRO_NVVM"))
print("NUMBAPRO_LIBDEVICE=", os.environ.get("NUMBAPRO_LIBDEVICE"))
print("LD_LIBRARY_PATH=", os.environ.get("LD_LIBRARY_PATH"))
ctypes.CDLL(os.environ["NUMBAPRO_NVVM"])
print("ctypes loaded libnvvm ok")
try:
    from numba import cuda
    print("numba cuda available:", cuda.is_available())
except Exception as e:
    print("numba cuda check failed:", repr(e))
PY
SANITY_STATUS=$?
if [[ $SANITY_STATUS -ne 0 ]]; then
  echo "SANITY_CHECK_FAILED exit_code=$SANITY_STATUS" >&2
  exit 20
fi
echo "SANITY_CHECK_PASSED"

if [[ -z "$CFG_FILE" || -z "$CKPT" ]]; then
  echo "Missing --cfg-file or --ckpt; sanity check completed without tools/test.py." >&2
  exit 0
fi

echo "=== OpenPCDet tools/test.py eval ==="
cd "$OPENPCDET_ROOT" || exit 12
PYTHONPATH=. "$PYTHON_BIN" - <<PY
import os
import runpy
import sys
import torch

root = os.getcwd()
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "tools"))

orig_torch_load = torch.load
def trusted_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return orig_torch_load(*args, **kwargs)
torch.load = trusted_load

sys.argv = [
    "tools/test.py",
    "--cfg_file", r"$CFG_FILE",
    "--ckpt", r"$CKPT",
    "--extra_tag", r"$EXTRA_TAG",
    "--eval_tag", r"$EVAL_TAG",
    "--batch_size", r"$BATCH_SIZE",
    "--workers", "0",
    "--save_to_file",
]
runpy.run_path(os.path.join(root, "tools", "test.py"), run_name="__main__")
PY
TOOLS_STATUS=$?
if [[ $TOOLS_STATUS -ne 0 ]]; then
  echo "TOOLS_TEST_FAILED exit_code=$TOOLS_STATUS" >&2
  exit 30
fi
echo "TOOLS_TEST_COMPLETED"
