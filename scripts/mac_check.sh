#!/usr/bin/env bash
# Apple Silicon check: unit tests + Sudoku-easy (first 128) on mps and cpu, outputs must match.
# Usage (from the repo root): bash scripts/mac_check.sh
# Downloads ~165 MB of weights/data and creates .venv if it does not exist.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p eval

if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null; then
    uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt pytest
  else
    python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt pytest
  fi
fi
PY=.venv/bin/python
$PY -c "import pytest" 2>/dev/null || $PY -m pip install -q pytest

CHIP=$(sysctl -n machdep.cpu.brand_string)
OS=$(sw_vers -productVersion)
RAM=$(( $(sysctl -n hw.memsize) / 1073741824 ))
TORCH=$($PY -c "import torch; print(torch.__version__)")
DEV=$($PY -c "import sys; sys.path.insert(0, '.'); from utils.device import get_device; print(get_device())")
echo "== $CHIP, ${RAM} GB, macOS $OS, torch $TORCH, device $DEV"

TESTS=$($PY -m pytest tests -q 2>&1 | tail -1)
echo "== tests: $TESTS"

[ -f artifacts/sudoku/easy/generator.pt ] || .venv/bin/hf download hazyresearch/ncm-checkpoints --include "sudoku/easy/*" --local-dir artifacts >/dev/null
[ -f data/sudoku/easy/puzzles.pickle ] || .venv/bin/hf download hazyresearch/ncm-datasets --repo-type dataset --include "sudoku/easy/*" --local-dir data >/dev/null

run() {
  local out=$1; shift
  rm -rf "$out"
  local t0=$SECONDS
  env "$@" $PY evaluate.py --config config/eval/sudoku-easy.yaml --checkpoint-dir artifacts/sudoku/easy \
    --data-dir data/sudoku/easy --n 128 --out "$out" > "$out.txt" 2>&1 || {
      { echo "FAILED: $*"; cat "$out.txt"; tail -20 "$out"/shard_0.log 2>/dev/null; } >&2; return 1; }
  echo $((SECONDS - t0))
}
T_DEF=$(run eval/mac_check_default PYTORCH_ENABLE_MPS_FALLBACK=0) || exit 1
T_CPU=$(run eval/mac_check_cpu NCM_DEVICE=cpu) || exit 1

RESULT=$($PY - <<'PY'
import pickle, re
s = lambda f: re.search(r"(\d+)/128", open(f).read()).group(1)
a = pickle.load(open("eval/mac_check_default/shard_0.pkl", "rb"))["hyps"]
b = pickle.load(open("eval/mac_check_cpu/shard_0.pkl", "rb"))["hyps"]
same = sum(x == y for x, y in zip(a, b))
print(f"sudoku {s('eval/mac_check_default.txt')}/128, identical to cpu {same}/{len(a)}")
PY
)
echo
echo "RESULT | $CHIP | ${RAM} GB | macOS $OS | torch $TORCH | $DEV | tests: $TESTS | $RESULT | ${T_DEF}s (cpu ${T_CPU}s)"
