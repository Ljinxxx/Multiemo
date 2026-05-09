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
echo "Start IEMOCAP: final keep, CMCL disabled"
echo "Time: $(date)"
echo "=========================================="

python Train/TrainMultiEMO.py \
  --dataset 'IEMOCAP' \
  --batch_size 64 \
  --num_layers 6 \
  --num_epochs 100 \
  --SWFC_loss_param 0.4 \
  --HGR_loss_param 0.2 \
  --CE_loss_param 0.4 \
  --aux_loss_param 0.2 \
  --cmcl_loss_param 0 \
  --cmcl_temp_param 0.2 \
  --sample_weight_param 1.1 \
  --temp_param 0.8 \
  --focus_param 2.4 \
  --dropout_rate 0.1 \
  --dropout_rec 0.1 \
  2>&1 | tee Log/IEMOCAP/IEMOCAP_final_keep_cmcl0.put

echo "=========================================="
echo "IEMOCAP finished"
echo "Time: $(date)"
echo "=========================================="

echo "=========================================="
echo "Start MELD: Cross-modal Contrastive Loss"
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
  --cmcl_loss_param 0.02 \
  --cmcl_temp_param 0.5 \
  --meld_label_smoothing 0.05 \
  --sample_weight_param 1.2 \
  --temp_param 1.4 \
  --focus_param 2.0 \
  --dropout_rate 0 \
  --dropout_rec 0 \
  2>&1 | tee Log/MELD/MELD_cmcl_005_temp02.put

echo "=========================================="
echo "MELD finished"
echo "Time: $(date)"
echo "=========================================="

echo "=========================================="
echo "Collecting Best Results"
echo "=========================================="

grep "Best test f1" Log/IEMOCAP/IEMOCAP_final_keep_cmcl0.put > Log/Summary/cmcl_two_datasets_summary.txt || true
grep "Best test f1" Log/MELD/MELD_cmcl_005_temp02.put >> Log/Summary/cmcl_two_datasets_summary.txt || true

echo "Summary saved to Log/Summary/cmcl_two_datasets_summary.txt"
cat Log/Summary/cmcl_two_datasets_summary.txt

echo "All CMCL experiments finished!"
