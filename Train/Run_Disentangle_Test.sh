#!/bin/bash

set -o pipefail

# ============================================================
# Environment settings
# ============================================================

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p Log/IEMOCAP/Disentangle
mkdir -p Log/MELD/Disentangle
mkdir -p Log/Summary

echo "=========================================="
echo "Checking Python files"
echo "=========================================="

python -m py_compile Model/CrossTaskGNN.py
python -m py_compile Train/TrainMultiEMO.py
python -m py_compile Model/MultiEMO_Model.py
python -m py_compile Model/MultiAttn.py

echo "=========================================="
echo "Python files checked"
echo "=========================================="

# ============================================================
# IEMOCAP identity disentanglement test
# ============================================================

echo ""
echo "=========================================="
echo "Start IEMOCAP Identity Disentanglement Test"
echo "Time: $(date)"
echo "=========================================="

python Train/TrainMultiEMO.py \
  --dataset 'IEMOCAP' \
  --batch_size 32 \
  --num_layers 6 \
  --num_epochs 20 \
  --SWFC_loss_param 0.4 \
  --HGR_loss_param 0.2 \
  --CE_loss_param 0.4 \
  --aux_loss_param 0.2 \
  --cmcl_loss_param 0 \
  --cmcl_temp_param 0.5 \
  --sample_weight_param 1.1 \
  --temp_param 0.8 \
  --focus_param 2.4 \
  --dropout_rate 0.1 \
  --dropout_rec 0.1 \
  --use_graph_mtl 1 \
  --gnn_alpha 0.1 \
  --gnn_edge_mode speaker_temporal \
  --graph_emotion_loss_param 0.01 \
  --identity_loss_param 0.01 \
  --adv_identity_loss_param 0.005 \
  --ortho_loss_param 0.001 \
  --grl_lambda 1.0 \
  --seed 2023 \
  2>&1 | tee Log/IEMOCAP/Disentangle/IEMOCAP_disentangle_test_seed2023.put

echo "=========================================="
echo "IEMOCAP Identity Disentanglement Test Finished"
echo "Time: $(date)"
echo "=========================================="

# ============================================================
# MELD identity disentanglement test
# ============================================================

echo ""
echo "=========================================="
echo "Start MELD Identity Disentanglement Test"
echo "Time: $(date)"
echo "=========================================="

python Train/TrainMultiEMO.py \
  --dataset 'MELD' \
  --batch_size 128 \
  --num_layers 4 \
  --num_epochs 20 \
  --SWFC_loss_param 0.4 \
  --HGR_loss_param 0.1 \
  --CE_loss_param 0.5 \
  --aux_loss_param 0 \
  --cmcl_loss_param 0.03 \
  --cmcl_temp_param 0.5 \
  --meld_label_smoothing 0.05 \
  --sample_weight_param 1.2 \
  --temp_param 1.4 \
  --focus_param 2.0 \
  --dropout_rate 0 \
  --dropout_rec 0 \
  --use_graph_mtl 1 \
  --gnn_alpha 0 \
  --gnn_edge_mode temporal \
  --graph_emotion_loss_param 0 \
  --identity_loss_param 0.005 \
  --adv_identity_loss_param 0.002 \
  --ortho_loss_param 0 \
  --grl_lambda 1.0 \
  --seed 2023 \
  2>&1 | tee Log/MELD/Disentangle/MELD_disentangle_test_seed2023.put

echo "=========================================="
echo "MELD Identity Disentanglement Test Finished"
echo "Time: $(date)"
echo "=========================================="

# ============================================================
# Collect summary
# ============================================================

echo ""
echo "=========================================="
echo "Collecting Disentanglement Test Results"
echo "=========================================="

SUMMARY_FILE="Log/Summary/disentangle_test_summary.txt"

rm -f "$SUMMARY_FILE"

echo "===== IEMOCAP Identity Disentanglement Test =====" >> "$SUMMARY_FILE"
grep "Best test f1" Log/IEMOCAP/Disentangle/IEMOCAP_disentangle_test_seed2023.put >> "$SUMMARY_FILE" || true
echo "" >> "$SUMMARY_FILE"

echo "===== MELD Identity Disentanglement Test =====" >> "$SUMMARY_FILE"
grep "Best test f1" Log/MELD/Disentangle/MELD_disentangle_test_seed2023.put >> "$SUMMARY_FILE" || true
echo "" >> "$SUMMARY_FILE"

cat "$SUMMARY_FILE"

echo "=========================================="
echo "Summary saved to $SUMMARY_FILE"
echo "All identity disentanglement test experiments finished!"
echo "=========================================="
