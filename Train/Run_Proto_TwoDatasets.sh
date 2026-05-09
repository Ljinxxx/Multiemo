#!/bin/bash

set -e
set -o pipefail

# ==============================
# Environment settings
# ==============================
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p Log/IEMOCAP
mkdir -p Log/MELD
mkdir -p Log/Summary

echo "=========================================="
echo "Start IEMOCAP: final keep, proto=0"
echo "Time: $(date)"
echo "=========================================="

python Train/TrainMultiEMO.py \
  --dataset 'IEMOCAP' \
  --batch_size 64 \
  --num_layers 6 \
  --num_epochs 150 \
  --SWFC_loss_param 0.4 \
  --HGR_loss_param 0.2 \
  --CE_loss_param 0.4 \
  --aux_loss_param 0.2 \
  --proto_loss_param 0 \
  --sample_weight_param 1.1 \
  --temp_param 0.8 \
  --focus_param 2.4 \
  --dropout_rate 0.1 \
  --dropout_rec 0.1 \
  2>&1 | tee Log/IEMOCAP/IEMOCAP_final_keep_proto0.put

echo "=========================================="
echo "IEMOCAP finished"
echo "Time: $(date)"
echo "=========================================="

echo "=========================================="
echo "Start MELD: prototype loss 0.05"
echo "Time: $(date)"
echo "=========================================="

python Train/TrainMultiEMO.py \
  --dataset 'MELD' \
  --batch_size 200 \
  --num_layers 4 \
  --num_epochs 40 \
  --SWFC_loss_param 0.4 \
  --HGR_loss_param 0.1 \
  --CE_loss_param 0.5 \
  --aux_loss_param 0 \
  --proto_loss_param 0.05 \
  --sample_weight_param 1.2 \
  --temp_param 1.4 \
  --focus_param 2.0 \
  --dropout_rate 0 \
  --dropout_rec 0 \
  2>&1 | tee Log/MELD/MELD_proto_loss_005.put

echo "=========================================="
echo "MELD finished"
echo "Time: $(date)"
echo "=========================================="

echo "=========================================="
echo "Collecting Best Results"
echo "=========================================="

grep "Best test f1" Log/IEMOCAP/IEMOCAP_final_keep_proto0.put > Log/Summary/proto_two_datasets_summary.txt || true
grep "Best test f1" Log/MELD/MELD_proto_loss_005.put >> Log/Summary/proto_two_datasets_summary.txt || true

echo "Summary saved to Log/Summary/proto_two_datasets_summary.txt"
cat Log/Summary/proto_two_datasets_summary.txt

echo "All prototype experiments finished!"
