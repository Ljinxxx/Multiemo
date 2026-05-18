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

mkdir -p Log/IEMOCAP/GraphMTL
mkdir -p Log/MELD/GraphMTL
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
# Utility function
# ============================================================

run_exp () {
  EXP_NAME="$1"
  LOG_FILE="$2"
  shift 2

  echo ""
  echo "=========================================="
  echo "Start experiment: ${EXP_NAME}"
  echo "Log file: ${LOG_FILE}"
  echo "Time: $(date)"
  echo "=========================================="

  if [ -f "$LOG_FILE" ] && grep -q "Best test f1" "$LOG_FILE"; then
    echo "Skip finished experiment: ${EXP_NAME}"
    grep "Best test f1" "$LOG_FILE" || true
    return 0
  fi

  "$@" 2>&1 | tee "$LOG_FILE"
  EXIT_CODE=${PIPESTATUS[0]}

  echo "=========================================="
  echo "Finished experiment: ${EXP_NAME}"
  echo "Exit code: ${EXIT_CODE}"
  echo "Time: $(date)"
  echo "=========================================="

  return 0
}

# ============================================================
# IEMOCAP Graph MTL experiments
# Recommended safe batch sizes: 32 / 48
# ============================================================

echo ""
echo "=========================================="
echo "Running IEMOCAP Graph MTL experiments"
echo "=========================================="

run_exp \
  "IEMOCAP_gmtl_bs32_a010_id003_ortho0005" \
  "Log/IEMOCAP/GraphMTL/IEMOCAP_gmtl_bs32_a010_id003_ortho0005_seed2023.put" \
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
    --graph_emotion_loss_param 0.02 \
    --identity_loss_param 0.03 \
    --ortho_loss_param 0.005 \
    --seed 2023

run_exp \
  "IEMOCAP_gmtl_bs48_a010_id003_ortho0005" \
  "Log/IEMOCAP/GraphMTL/IEMOCAP_gmtl_bs48_a010_id003_ortho0005_seed2023.put" \
  python Train/TrainMultiEMO.py \
    --dataset 'IEMOCAP' \
    --batch_size 48 \
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
    --graph_emotion_loss_param 0.02 \
    --identity_loss_param 0.03 \
    --ortho_loss_param 0.005 \
    --seed 2023

run_exp \
  "IEMOCAP_gmtl_bs48_a005_id002_ortho0003" \
  "Log/IEMOCAP/GraphMTL/IEMOCAP_gmtl_bs48_a005_id002_ortho0003_seed2023.put" \
  python Train/TrainMultiEMO.py \
    --dataset 'IEMOCAP' \
    --batch_size 48 \
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
    --gnn_alpha 0.05 \
    --gnn_edge_mode speaker_temporal \
    --graph_emotion_loss_param 0.01 \
    --identity_loss_param 0.02 \
    --ortho_loss_param 0.003 \
    --seed 2023

run_exp \
  "IEMOCAP_gmtl_bs48_a010_id001_ortho0000" \
  "Log/IEMOCAP/GraphMTL/IEMOCAP_gmtl_bs48_a010_id001_ortho0000_seed2023.put" \
  python Train/TrainMultiEMO.py \
    --dataset 'IEMOCAP' \
    --batch_size 48 \
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
    --graph_emotion_loss_param 0.01 \
    --identity_loss_param 0.01 \
    --ortho_loss_param 0 \
    --seed 2023

# ============================================================
# MELD Graph MTL experiments
# Recommended safe batch sizes: 100 / 128
# For MELD, temporal-only graph is safer.
# ============================================================

echo ""
echo "=========================================="
echo "Running MELD Graph MTL experiments"
echo "=========================================="

run_exp \
  "MELD_gmtl_bs100_a002_id002_ortho0003" \
  "Log/MELD/GraphMTL/MELD_gmtl_bs100_a002_id002_ortho0003_seed2023.put" \
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
    --graph_emotion_loss_param 0.01 \
    --identity_loss_param 0.02 \
    --ortho_loss_param 0.003 \
    --seed 2023

run_exp \
  "MELD_gmtl_bs128_a002_id002_ortho0003" \
  "Log/MELD/GraphMTL/MELD_gmtl_bs128_a002_id002_ortho0003_seed2023.put" \
  python Train/TrainMultiEMO.py \
    --dataset 'MELD' \
    --batch_size 128 \
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
    --graph_emotion_loss_param 0.01 \
    --identity_loss_param 0.02 \
    --ortho_loss_param 0.003 \
    --seed 2023

run_exp \
  "MELD_gmtl_bs128_a001_id001_ortho0000" \
  "Log/MELD/GraphMTL/MELD_gmtl_bs128_a001_id001_ortho0000_seed2023.put" \
  python Train/TrainMultiEMO.py \
    --dataset 'MELD' \
    --batch_size 128 \
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
    --gnn_alpha 0.01 \
    --gnn_edge_mode temporal \
    --graph_emotion_loss_param 0.005 \
    --identity_loss_param 0.01 \
    --ortho_loss_param 0 \
    --seed 2023

run_exp \
  "MELD_gmtl_bs128_a005_id001_ortho0000" \
  "Log/MELD/GraphMTL/MELD_gmtl_bs128_a005_id001_ortho0000_seed2023.put" \
  python Train/TrainMultiEMO.py \
    --dataset 'MELD' \
    --batch_size 128 \
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
    --gnn_alpha 0.05 \
    --gnn_edge_mode temporal \
    --graph_emotion_loss_param 0.005 \
    --identity_loss_param 0.01 \
    --ortho_loss_param 0 \
    --seed 2023

run_exp \
  "MELD_baseline_cmcl_no_graph" \
  "Log/MELD/GraphMTL/MELD_baseline_cmcl_no_graph_seed2023.put" \
  python Train/TrainMultiEMO.py \
    --dataset 'MELD' \
    --batch_size 200 \
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
    --use_graph_mtl 0 \
    --gnn_alpha 0 \
    --gnn_edge_mode temporal \
    --graph_emotion_loss_param 0 \
    --identity_loss_param 0 \
    --ortho_loss_param 0 \
    --seed 2023

# ============================================================
# Collect summary
# ============================================================

echo ""
echo "=========================================="
echo "Collecting Graph MTL Summary"
echo "=========================================="

SUMMARY_TXT="Log/Summary/graph_mtl_multi_experiments_summary.txt"
SUMMARY_CSV="Log/Summary/graph_mtl_multi_experiments_summary.csv"

rm -f "$SUMMARY_TXT"
rm -f "$SUMMARY_CSV"

echo "===== IEMOCAP Graph MTL Results =====" >> "$SUMMARY_TXT"

for f in Log/IEMOCAP/GraphMTL/*.put
do
  echo "$(basename "$f")" >> "$SUMMARY_TXT"
  grep "Best test f1" "$f" >> "$SUMMARY_TXT" || echo "No Best test f1 found" >> "$SUMMARY_TXT"
  echo "" >> "$SUMMARY_TXT"
done

echo "===== MELD Graph MTL Results =====" >> "$SUMMARY_TXT"

for f in Log/MELD/GraphMTL/*.put
do
  echo "$(basename "$f")" >> "$SUMMARY_TXT"
  grep "Best test f1" "$f" >> "$SUMMARY_TXT" || echo "No Best test f1 found" >> "$SUMMARY_TXT"
  echo "" >> "$SUMMARY_TXT"
done

python - <<'PY'
import os
import re
import glob
import csv

rows = []

for dataset, pattern in [
    ("IEMOCAP", "Log/IEMOCAP/GraphMTL/*.put"),
    ("MELD", "Log/MELD/GraphMTL/*.put"),
]:
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        m = re.search(r"Best test f1:\s*([0-9.]+)\s*at epoch\s*([0-9]+)", text)

        if m:
            best_f1 = float(m.group(1))
            best_epoch = int(m.group(2))
            status = "finished"
        else:
            best_f1 = ""
            best_epoch = ""
            status = "failed_or_unfinished"

        rows.append({
            "dataset": dataset,
            "file": name,
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "status": status,
        })

csv_path = "Log/Summary/graph_mtl_multi_experiments_summary.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["dataset", "file", "best_f1", "best_epoch", "status"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("CSV saved to", csv_path)

print("\n===== Sorted Finished Results =====")
finished = [r for r in rows if r["status"] == "finished"]
finished.sort(key=lambda x: float(x["best_f1"]), reverse=True)

for r in finished:
    print(f'{r["dataset"]}\t{r["best_f1"]}\tepoch {r["best_epoch"]}\t{r["file"]}')
PY

cat "$SUMMARY_TXT"

echo "=========================================="
echo "Summary saved to:"
echo "$SUMMARY_TXT"
echo "$SUMMARY_CSV"
echo "All Graph MTL multi experiments finished!"
echo "=========================================="
