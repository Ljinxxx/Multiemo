#!/bin/bash

set -e
set -o pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p Log/MELD
mkdir -p Log/Summary

run_meld () {
  NAME=$1
  CMCL_W=$2
  CMCL_T=$3

  echo "=========================================="
  echo "Start MELD ${NAME}: cmcl=${CMCL_W}, temp=${CMCL_T}"
  echo "Time: $(date)"
  echo "=========================================="

  python Train/TrainMultiEMO.py \
    --dataset 'MELD' \
    --batch_size 200 \
    --num_layers 4 \
    --num_epochs 25 \
    --SWFC_loss_param 0.4 \
    --HGR_loss_param 0.1 \
    --CE_loss_param 0.5 \
    --aux_loss_param 0 \
    --cmcl_loss_param ${CMCL_W} \
    --cmcl_temp_param ${CMCL_T} \
    --meld_label_smoothing 0.05 \
    --sample_weight_param 1.2 \
    --temp_param 1.4 \
    --focus_param 2.0 \
    --dropout_rate 0 \
    --dropout_rec 0 \
    2>&1 | tee Log/MELD/MELD_${NAME}.put

  echo "Finished MELD ${NAME}"
  echo "Time: $(date)"
}

run_meld "cmcl0015_temp05" 0.015 0.5
run_meld "cmcl0025_temp05" 0.025 0.5
run_meld "cmcl003_temp05" 0.03 0.5   #最优
run_meld "cmcl002_temp07" 0.035 0.7


echo "=========================================="
echo "Collecting MELD CMCL Search Results"
echo "=========================================="

grep "Best test f1" Log/MELD/MELD_cmcl0015_temp05.put > Log/Summary/meld_cmcl_search_summary.txt || true
grep "Best test f1" Log/MELD/MELD_cmcl0025_temp05.put >> Log/Summary/meld_cmcl_search_summary.txt || true
grep "Best test f1" Log/MELD/MELD_cmcl003_temp05.put >> Log/Summary/meld_cmcl_search_summary.txt || true
grep "Best test f1" Log/MELD/MELD_cmcl002_temp07.put >> Log/Summary/meld_cmcl_search_summary.txt || true

echo "Summary saved to Log/Summary/meld_cmcl_search_summary.txt"
cat Log/Summary/meld_cmcl_search_summary.txt

echo "All MELD CMCL search experiments finished!"
