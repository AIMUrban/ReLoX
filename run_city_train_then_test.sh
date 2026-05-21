#!/usr/bin/env bash
set -euo pipefail

# Train on one city, then zero-shot test on 3 cities.
#
# Example:
#   ./run_city_train_then_test.sh --train kumamoto --tests kumamoto hiroshima sapporo --gpu 0 --seed 42

GPUS="0"
SEED="42"
DESC=""
TRAIN_CITY="kumamoto"
TEST_CITIES=("kumamoto" "hiroshima" "sapporo")

EPOCHS=10
BSZ=256
LR=1e-3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPUS="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --desc) DESC="$2"; shift 2;;
    --train) TRAIN_CITY="$2"; shift 2;;
    --tests)
      shift 1
      TEST_CITIES=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        TEST_CITIES+=("$1"); shift 1
      done
      ;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch_size) BSZ="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    *) echo "Unknown argument: $1"; exit 1;;
  esac
done

export PYTHONHASHSEED=$SEED
export CUDA_VISIBLE_DEVICES=${GPUS}

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
TRAIN_ROOT="dataset/${TRAIN_CITY}"
TEST_ROOTS=()
for c in "${TEST_CITIES[@]}"; do
  TEST_ROOTS+=("dataset/${c}")
done

echo "[run_city_train_then_test.sh] train=${TRAIN_ROOT} tests=${TEST_ROOTS[*]} gpus=${CUDA_VISIBLE_DEVICES} seed=${SEED} desc='${DESC}'"

COMMON_ARGS=(
  --data_root "$TRAIN_ROOT"
  --test_data_roots "${TEST_ROOTS[@]}"
  --epochs "$EPOCHS"
  --batch_size "$BSZ"
  --lr "$LR"
  --seed "$SEED"
  --lambda_rel 1
  --lambda_abs 1
  --rel_size 8
  --match_dim 128
  --drop_r_p 0.5
)

if [ -n "$DESC" ]; then
  COMMON_ARGS+=(--desc "$DESC")
fi

if [ "$NUM_GPUS" -gt 1 ]; then
  torchrun --nproc_per_node="$NUM_GPUS" main.py "${COMMON_ARGS[@]}"
else
  python main.py "${COMMON_ARGS[@]}"
fi
