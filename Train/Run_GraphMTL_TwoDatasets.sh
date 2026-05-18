#!/bin/bash

set -e
set -o pipefail

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
echo "Checking Python files"
echo "=========================================="

python -m py_compile Model/CrossTaskGNN.py
python -m py_compile Train/TrainMultiEMO.py
python -m py_compile Model/MultiEMO_Model.py
python -m py_compile Model/MultiAttn.py

echo "=========================================="
echo "Start IEMOCAP: Graph Multi-task Emotion + Identity"
echo "Time: $(date)"
echo "=========================================="

python Train/TrainMultiEMO.py \
  --dataset 'IEMOCAP' \
  --batch_size 32 \
  --num_layers 6 \
  --num_epochs 100 \
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
  --graph_emotion_loss_param 0.03 \
  --identity_loss_param 0.05 \
  --ortho_loss_param 0.01 \
  --seed 2023 \
  2>&1 | tee Log/IEMOCAP/IEMOCAP_graph_mtl_id_alpha01_seed2023.put

echo "=========================================="
echo "IEMOCAP finished"
echo "Time: $(date)"
echo "=========================================="

echo "=========================================="
echo "Start MELD: Graph Multi-task Emotion + Identity"
echo "Time: $(date)"
echo "=========================================="

python Train/TrainMultiEMO.py \
  --dataset 'MELD' \
  --batch_size 100 \
  --num_layers 4 \
  --num_epochs 40 \
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
  --gnn_alpha 0.02 \
  --gnn_edge_mode temporal \
  --graph_emotion_loss_param 0.02 \
  --identity_loss_param 0.03 \
  --ortho_loss_param 0.005 \
  --seed 2023 \
  2>&1 | tee Log/MELD/MELD_graph_mtl_id_temporal_alpha002_seed2023.put

echo "=========================================="
echo "MELD finished"
echo "Time: $(date)"
echo "=========================================="

echo "=========================================="
echo "Collecting Graph MTL Results"
echo "=========================================="

SUMMARY_FILE="Log/Summary/graph_mtl_two_datasets_summary.txt"
rm -f "$SUMMARY_FILE"

echo "===== IEMOCAP Graph MTL =====" >> "$SUMMARY_FILE"
grep "Best test f1" Log/IEMOCAP/IEMOCAP_graph_mtl_id_alpha01_seed2023.put >> "$SUMMARY_FILE" || true
echo "" >> "$SUMMARY_FILE"

echo "===== MELD Graph MTL =====" >> "$SUMMARY_FILE"
grep "Best test f1" Log/MELD/MELD_graph_mtl_id_temporal_alpha002_seed2023.put >> "$SUMMARY_FILE" || true
echo "" >> "$SUMMARY_FILE"

cat "$SUMMARY_FILE"

echo "=========================================="
echo "Summary saved to $SUMMARY_FILE"
echo "All Graph MTL experiments finished!"
echo "=========================================="
