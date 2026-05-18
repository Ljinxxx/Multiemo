#!/bin/bash

set -o pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p Log/IEMOCAP/GraphMTL_Next
mkdir -p Log/MELD/GraphMTL_Next
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
# Part 1: IEMOCAP Graph MTL stability experiments
# Current best: around 73.25 at epoch 98
# Next step: extend to 150 epochs and run multiple seeds
# ============================================================

echo ""
echo "=========================================="
echo "Running IEMOCAP Graph MTL stability experiments"
echo "=========================================="

for SEED in 2023 2024 2025
do
  run_exp \
    "IEMOCAP_GraphMTL_best_150ep_seed${SEED}" \
    "Log/IEMOCAP/GraphMTL_Next/IEMOCAP_graph_mtl_best_150ep_seed${SEED}.put" \
    python Train/TrainMultiEMO.py \
      --dataset 'IEMOCAP' \
      --batch_size 32 \
      --num_layers 6 \
      --num_epochs 150 \
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
      --seed ${SEED}
done

# A lighter identity setting for IEMOCAP
run_exp \
  "IEMOCAP_GraphMTL_light_identity_150ep_seed2023" \
  "Log/IEMOCAP/GraphMTL_Next/IEMOCAP_graph_mtl_light_identity_150ep_seed2023.put" \
  python Train/TrainMultiEMO.py \
    --dataset 'IEMOCAP' \
    --batch_size 32 \
    --num_layers 6 \
    --num_epochs 150 \
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
# Part 2: MELD auxiliary Graph MTL experiments
# Goal: avoid damaging main emotion classifier
# Key idea: gnn_alpha=0
# ============================================================

echo ""
echo "=========================================="
echo "Running MELD auxiliary Graph MTL experiments"
echo "=========================================="

# MELD baseline: ordinary CMCL, no Graph MTL
run_exp \
  "MELD_CMCL_baseline_no_graph_seed2023" \
  "Log/MELD/GraphMTL_Next/MELD_cmcl_baseline_no_graph_seed2023.put" \
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

# MELD auxiliary mode 1:
# Graph emotion + identity auxiliary, but no residual perturbation
run_exp \
  "MELD_aux_graph_identity_no_residual_seed2023" \
  "Log/MELD/GraphMTL_Next/MELD_aux_graph_identity_no_residual_seed2023.put" \
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
    --gnn_alpha 0 \
    --gnn_edge_mode temporal \
    --graph_emotion_loss_param 0.005 \
    --identity_loss_param 0.005 \
    --ortho_loss_param 0 \
    --seed 2023

# MELD auxiliary mode 2:
# Only identity auxiliary, no graph emotion loss
run_exp \
  "MELD_aux_identity_only_no_residual_seed2023" \
  "Log/MELD/GraphMTL_Next/MELD_aux_identity_only_no_residual_seed2023.put" \
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
    --gnn_alpha 0 \
    --gnn_edge_mode temporal \
    --graph_emotion_loss_param 0 \
    --identity_loss_param 0.005 \
    --ortho_loss_param 0 \
    --seed 2023

# MELD auxiliary mode 3:
# Only graph emotion auxiliary, no identity loss
run_exp \
  "MELD_aux_graph_emotion_only_no_residual_seed2023" \
  "Log/MELD/GraphMTL_Next/MELD_aux_graph_emotion_only_no_residual_seed2023.put" \
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
    --gnn_alpha 0 \
    --gnn_edge_mode temporal \
    --graph_emotion_loss_param 0.005 \
    --identity_loss_param 0 \
    --ortho_loss_param 0 \
    --seed 2023

# MELD weak residual mode:
# Very weak residual + identity only
run_exp \
  "MELD_weak_residual_identity_seed2023" \
  "Log/MELD/GraphMTL_Next/MELD_weak_residual_identity_seed2023.put" \
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
    --gnn_alpha 0.005 \
    --gnn_edge_mode temporal \
    --graph_emotion_loss_param 0 \
    --identity_loss_param 0.005 \
    --ortho_loss_param 0 \
    --seed 2023

# ============================================================
# Summary
# ============================================================

echo ""
echo "=========================================="
echo "Collecting Summary"
echo "=========================================="

SUMMARY_TXT="Log/Summary/next_graph_mtl_improve_summary.txt"
SUMMARY_CSV="Log/Summary/next_graph_mtl_improve_summary.csv"

rm -f "$SUMMARY_TXT"
rm -f "$SUMMARY_CSV"

echo "===== IEMOCAP Next Graph MTL Results =====" >> "$SUMMARY_TXT"

for f in Log/IEMOCAP/GraphMTL_Next/*.put
do
  echo "$(basename "$f")" >> "$SUMMARY_TXT"
  grep "Best test f1" "$f" >> "$SUMMARY_TXT" || echo "No Best test f1 found" >> "$SUMMARY_TXT"
  echo "" >> "$SUMMARY_TXT"
done

echo "===== MELD Next Graph MTL Results =====" >> "$SUMMARY_TXT"

for f in Log/MELD/GraphMTL_Next/*.put
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
    ("IEMOCAP", "Log/IEMOCAP/GraphMTL_Next/*.put"),
    ("MELD", "Log/MELD/GraphMTL_Next/*.put"),
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

csv_path = "Log/Summary/next_graph_mtl_improve_summary.csv"

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
echo "All next Graph MTL experiments finished!"
echo "=========================================="
