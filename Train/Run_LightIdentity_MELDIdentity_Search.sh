#!/bin/bash

set -o pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p Log/IEMOCAP/LightIdentity
mkdir -p Log/MELD/IdentityOnly
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
# Part 1: IEMOCAP light identity multi-seed
# Current best:
# graph_emotion_loss_param = 0.01
# identity_loss_param = 0.01
# ortho_loss_param = 0
# ============================================================

echo ""
echo "=========================================="
echo "Running IEMOCAP light identity multi-seed"
echo "=========================================="

for SEED in 2023 2024 2025 2026 2027
do
  run_exp \
    "IEMOCAP_light_identity_seed${SEED}" \
    "Log/IEMOCAP/LightIdentity/IEMOCAP_light_identity_seed${SEED}.put" \
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
      --seed ${SEED}
done

# ============================================================
# Part 2: MELD identity-only auxiliary search
# Do not use graph emotion loss.
# Do not use ortho loss.
# Search identity loss and very weak residual.
# ============================================================

echo ""
echo "=========================================="
echo "Running MELD identity-only auxiliary search"
echo "=========================================="

# Baseline: CMCL no graph
run_exp \
  "MELD_baseline_cmcl_no_graph_seed2023" \
  "Log/MELD/IdentityOnly/MELD_baseline_cmcl_no_graph_seed2023.put" \
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

# identity = 0.002, no residual
run_exp \
  "MELD_identity_only_alpha0_id0002_seed2023" \
  "Log/MELD/IdentityOnly/MELD_identity_only_alpha0_id0002_seed2023.put" \
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
    --identity_loss_param 0.002 \
    --ortho_loss_param 0 \
    --seed 2023

# identity = 0.005, no residual
run_exp \
  "MELD_identity_only_alpha0_id0005_seed2023" \
  "Log/MELD/IdentityOnly/MELD_identity_only_alpha0_id0005_seed2023.put" \
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

# identity = 0.01, no residual
run_exp \
  "MELD_identity_only_alpha0_id001_seed2023" \
  "Log/MELD/IdentityOnly/MELD_identity_only_alpha0_id001_seed2023.put" \
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
    --identity_loss_param 0.01 \
    --ortho_loss_param 0 \
    --seed 2023

# weak residual alpha=0.002, identity=0.005
run_exp \
  "MELD_identity_only_alpha0002_id0005_seed2023" \
  "Log/MELD/IdentityOnly/MELD_identity_only_alpha0002_id0005_seed2023.put" \
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
    --gnn_alpha 0.002 \
    --gnn_edge_mode temporal \
    --graph_emotion_loss_param 0 \
    --identity_loss_param 0.005 \
    --ortho_loss_param 0 \
    --seed 2023

# weak residual alpha=0.005, identity=0.005
run_exp \
  "MELD_identity_only_alpha0005_id0005_seed2023" \
  "Log/MELD/IdentityOnly/MELD_identity_only_alpha0005_id0005_seed2023.put" \
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

SUMMARY_TXT="Log/Summary/light_identity_meld_identity_search_summary.txt"
SUMMARY_CSV="Log/Summary/light_identity_meld_identity_search_summary.csv"

rm -f "$SUMMARY_TXT"
rm -f "$SUMMARY_CSV"

echo "===== IEMOCAP Light Identity Results =====" >> "$SUMMARY_TXT"

for f in Log/IEMOCAP/LightIdentity/*.put
do
  echo "$(basename "$f")" >> "$SUMMARY_TXT"
  grep "Best test f1" "$f" >> "$SUMMARY_TXT" || echo "No Best test f1 found" >> "$SUMMARY_TXT"
  echo "" >> "$SUMMARY_TXT"
done

echo "===== MELD Identity-only Results =====" >> "$SUMMARY_TXT"

for f in Log/MELD/IdentityOnly/*.put
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
    ("IEMOCAP", "Log/IEMOCAP/LightIdentity/*.put"),
    ("MELD", "Log/MELD/IdentityOnly/*.put"),
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

csv_path = "Log/Summary/light_identity_meld_identity_search_summary.csv"

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
echo "All light-identity and MELD identity-only experiments finished!"
echo "=========================================="
