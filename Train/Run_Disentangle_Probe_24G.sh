#!/bin/bash

set -o pipefail

# ============================================================
# 24G GPU optimized settings
# ============================================================

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

SEED=2023

IEMOCAP_BATCH_SIZE=48
MELD_BATCH_SIZE=160

IEMOCAP_EPOCHS=150
MELD_EPOCHS=40

PROBE_EPOCHS=100
PROBE_BATCH_SIZE=1024
PROBE_LR=0.001

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p Log/IEMOCAP/Disentangle24G
mkdir -p Log/MELD/Disentangle24G
mkdir -p Log/IEMOCAP/DisentangleAblation24G
mkdir -p Log/MELD/DisentangleAblation24G
mkdir -p Log/Summary
mkdir -p Checkpoints/IEMOCAP
mkdir -p Checkpoints/MELD

echo "=========================================="
echo "Checking Python files"
echo "=========================================="

python -m py_compile Model/CrossTaskGNN.py || exit 1
python -m py_compile Train/TrainMultiEMO.py || exit 1
python -m py_compile Train/ProbeIdentityLeakage.py || exit 1
python -m py_compile Model/MultiEMO_Model.py || exit 1
python -m py_compile Model/MultiAttn.py || exit 1

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

run_probe () {
  EXP_NAME="$1"
  CHECKPOINT="$2"
  LOG_FILE="$3"
  OUTPUT_FILE="$4"
  DATA_BATCH_SIZE="$5"

  echo ""
  echo "=========================================="
  echo "Start probe: ${EXP_NAME}"
  echo "Checkpoint: ${CHECKPOINT}"
  echo "Log file: ${LOG_FILE}"
  echo "Output file: ${OUTPUT_FILE}"
  echo "Time: $(date)"
  echo "=========================================="

  if [ ! -f "$CHECKPOINT" ]; then
    echo "WARNING: checkpoint not found: ${CHECKPOINT}"
    echo "Skip probe: ${EXP_NAME}"
    return 0
  fi

  if [ -f "$OUTPUT_FILE" ] && grep -q "Identity Leakage Probe Results" "$OUTPUT_FILE"; then
    echo "Skip finished probe: ${EXP_NAME}"
    cat "$OUTPUT_FILE"
    return 0
  fi

  python Train/ProbeIdentityLeakage.py \
    --checkpoint "$CHECKPOINT" \
    --data_batch_size "$DATA_BATCH_SIZE" \
    --probe_epochs "$PROBE_EPOCHS" \
    --probe_lr "$PROBE_LR" \
    --probe_batch_size "$PROBE_BATCH_SIZE" \
    --seed "$SEED" \
    --output "$OUTPUT_FILE" \
    2>&1 | tee "$LOG_FILE"

  echo "=========================================="
  echo "Finished probe: ${EXP_NAME}"
  echo "Time: $(date)"
  echo "=========================================="
}

# ============================================================
# Part 1: Train main disentanglement models and run probes
# ============================================================

echo ""
echo "=========================================="
echo "Part 1: Train + probe identity disentanglement"
echo "=========================================="

run_exp \
  "IEMOCAP_main_disentangle_24G" \
  "Log/IEMOCAP/Disentangle24G/IEMOCAP_main_disentangle_seed${SEED}.put" \
  python Train/TrainMultiEMO.py \
    --dataset 'IEMOCAP' \
    --batch_size "$IEMOCAP_BATCH_SIZE" \
    --num_layers 6 \
    --num_epochs "$IEMOCAP_EPOCHS" \
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
    --seed "$SEED"

if [ -f Checkpoints/IEMOCAP/best_disentangle_checkpoint.pt ]; then
  cp Checkpoints/IEMOCAP/best_disentangle_checkpoint.pt \
     Checkpoints/IEMOCAP/best_disentangle_checkpoint_main_24G_seed${SEED}.pt
fi

run_probe \
  "IEMOCAP_identity_leakage_probe_24G" \
  "Checkpoints/IEMOCAP/best_disentangle_checkpoint_main_24G_seed${SEED}.pt" \
  "Log/IEMOCAP/Disentangle24G/IEMOCAP_identity_leakage_probe_seed${SEED}.put" \
  "Log/Summary/IEMOCAP_identity_leakage_probe_24G.txt" \
  "$IEMOCAP_BATCH_SIZE"

run_exp \
  "MELD_main_disentangle_24G" \
  "Log/MELD/Disentangle24G/MELD_main_disentangle_seed${SEED}.put" \
  python Train/TrainMultiEMO.py \
    --dataset 'MELD' \
    --batch_size "$MELD_BATCH_SIZE" \
    --num_layers 4 \
    --num_epochs "$MELD_EPOCHS" \
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
    --seed "$SEED"

if [ -f Checkpoints/MELD/best_disentangle_checkpoint.pt ]; then
  cp Checkpoints/MELD/best_disentangle_checkpoint.pt \
     Checkpoints/MELD/best_disentangle_checkpoint_main_24G_seed${SEED}.pt
fi

run_probe \
  "MELD_identity_leakage_probe_24G" \
  "Checkpoints/MELD/best_disentangle_checkpoint_main_24G_seed${SEED}.pt" \
  "Log/MELD/Disentangle24G/MELD_identity_leakage_probe_seed${SEED}.put" \
  "Log/Summary/MELD_identity_leakage_probe_24G.txt" \
  "$MELD_BATCH_SIZE"

# ============================================================
# Part 2: Ablation experiments
# ============================================================

echo ""
echo "=========================================="
echo "Part 2: Run disentanglement ablation"
echo "=========================================="

run_iemocap_ablation () {
  EXP_NAME="$1"
  USE_GRAPH_MTL="$2"
  GNN_ALPHA="$3"
  GRAPH_EMO="$4"
  ID_LOSS="$5"
  ADV_ID="$6"
  ORTHO="$7"

  run_exp \
    "${EXP_NAME}" \
    "Log/IEMOCAP/DisentangleAblation24G/${EXP_NAME}_seed${SEED}.put" \
    python Train/TrainMultiEMO.py \
      --dataset 'IEMOCAP' \
      --batch_size "$IEMOCAP_BATCH_SIZE" \
      --num_layers 6 \
      --num_epochs "$IEMOCAP_EPOCHS" \
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
      --use_graph_mtl "${USE_GRAPH_MTL}" \
      --gnn_alpha "${GNN_ALPHA}" \
      --gnn_edge_mode speaker_temporal \
      --graph_emotion_loss_param "${GRAPH_EMO}" \
      --identity_loss_param "${ID_LOSS}" \
      --adv_identity_loss_param "${ADV_ID}" \
      --ortho_loss_param "${ORTHO}" \
      --grl_lambda 1.0 \
      --seed "$SEED"
}

run_meld_ablation () {
  EXP_NAME="$1"
  USE_GRAPH_MTL="$2"
  GNN_ALPHA="$3"
  GRAPH_EMO="$4"
  ID_LOSS="$5"
  ADV_ID="$6"
  ORTHO="$7"

  run_exp \
    "${EXP_NAME}" \
    "Log/MELD/DisentangleAblation24G/${EXP_NAME}_seed${SEED}.put" \
    python Train/TrainMultiEMO.py \
      --dataset 'MELD' \
      --batch_size "$MELD_BATCH_SIZE" \
      --num_layers 4 \
      --num_epochs "$MELD_EPOCHS" \
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
      --use_graph_mtl "${USE_GRAPH_MTL}" \
      --gnn_alpha "${GNN_ALPHA}" \
      --gnn_edge_mode temporal \
      --graph_emotion_loss_param "${GRAPH_EMO}" \
      --identity_loss_param "${ID_LOSS}" \
      --adv_identity_loss_param "${ADV_ID}" \
      --ortho_loss_param "${ORTHO}" \
      --grl_lambda 1.0 \
      --seed "$SEED"
}

echo ""
echo "=========================================="
echo "IEMOCAP ablation"
echo "=========================================="

run_iemocap_ablation "IEMOCAP_A_no_graph_24G"              0 0    0    0    0     0
run_iemocap_ablation "IEMOCAP_B_gnn_residual_only_24G"     1 0.1  0    0    0     0
run_iemocap_ablation "IEMOCAP_C_graph_emotion_only_24G"    1 0.1  0.01 0    0     0
run_iemocap_ablation "IEMOCAP_D_light_identity_24G"        1 0.1  0.01 0.01 0     0
run_iemocap_ablation "IEMOCAP_E_identity_adv_24G"          1 0.1  0.01 0.01 0.005 0
run_iemocap_ablation "IEMOCAP_F_full_disentangle_24G"      1 0.1  0.01 0.01 0.005 0.001

echo ""
echo "=========================================="
echo "MELD ablation"
echo "=========================================="

run_meld_ablation "MELD_A_no_graph_24G"                   0 0     0 0     0     0
run_meld_ablation "MELD_B_identity_only_24G"              1 0     0 0.005 0     0
run_meld_ablation "MELD_C_adv_only_24G"                   1 0     0 0     0.002 0
run_meld_ablation "MELD_D_identity_adv_24G"               1 0     0 0.005 0.002 0
run_meld_ablation "MELD_E_identity_adv_weak_residual_24G" 1 0.002 0 0.005 0.002 0

# ============================================================
# Part 3: Summary
# ============================================================

echo ""
echo "=========================================="
echo "Collecting final summary"
echo "=========================================="

SUMMARY_TXT="Log/Summary/disentangle_probe_and_ablation_24G_summary.txt"
SUMMARY_CSV="Log/Summary/disentangle_probe_and_ablation_24G_summary.csv"

rm -f "$SUMMARY_TXT"
rm -f "$SUMMARY_CSV"

echo "===== 24G Config =====" >> "$SUMMARY_TXT"
echo "IEMOCAP batch_size: $IEMOCAP_BATCH_SIZE" >> "$SUMMARY_TXT"
echo "MELD batch_size: $MELD_BATCH_SIZE" >> "$SUMMARY_TXT"
echo "Probe batch_size: $PROBE_BATCH_SIZE" >> "$SUMMARY_TXT"
echo "Seed: $SEED" >> "$SUMMARY_TXT"
echo "" >> "$SUMMARY_TXT"

echo "===== Identity Leakage Probe: IEMOCAP =====" >> "$SUMMARY_TXT"
cat Log/Summary/IEMOCAP_identity_leakage_probe_24G.txt >> "$SUMMARY_TXT" 2>/dev/null || echo "No IEMOCAP probe result found" >> "$SUMMARY_TXT"
echo "" >> "$SUMMARY_TXT"

echo "===== Identity Leakage Probe: MELD =====" >> "$SUMMARY_TXT"
cat Log/Summary/MELD_identity_leakage_probe_24G.txt >> "$SUMMARY_TXT" 2>/dev/null || echo "No MELD probe result found" >> "$SUMMARY_TXT"
echo "" >> "$SUMMARY_TXT"

echo "===== IEMOCAP Main Disentangle =====" >> "$SUMMARY_TXT"
grep "Best test f1" Log/IEMOCAP/Disentangle24G/*.put >> "$SUMMARY_TXT" 2>/dev/null || true
echo "" >> "$SUMMARY_TXT"

echo "===== MELD Main Disentangle =====" >> "$SUMMARY_TXT"
grep "Best test f1" Log/MELD/Disentangle24G/*.put >> "$SUMMARY_TXT" 2>/dev/null || true
echo "" >> "$SUMMARY_TXT"

echo "===== IEMOCAP Disentanglement Ablation =====" >> "$SUMMARY_TXT"

for f in Log/IEMOCAP/DisentangleAblation24G/*.put
do
  echo "$(basename "$f")" >> "$SUMMARY_TXT"
  grep "Best test f1" "$f" >> "$SUMMARY_TXT" || echo "No Best test f1 found" >> "$SUMMARY_TXT"
  echo "" >> "$SUMMARY_TXT"
done

echo "===== MELD Disentanglement Ablation =====" >> "$SUMMARY_TXT"

for f in Log/MELD/DisentangleAblation24G/*.put
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

patterns = [
    ("IEMOCAP_MAIN", "Log/IEMOCAP/Disentangle24G/*.put"),
    ("MELD_MAIN", "Log/MELD/Disentangle24G/*.put"),
    ("IEMOCAP_ABLATION", "Log/IEMOCAP/DisentangleAblation24G/*.put"),
    ("MELD_ABLATION", "Log/MELD/DisentangleAblation24G/*.put"),
]

for dataset, pattern in patterns:
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
            "group": dataset,
            "file": name,
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "status": status,
        })

csv_path = "Log/Summary/disentangle_probe_and_ablation_24G_summary.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["group", "file", "best_f1", "best_epoch", "status"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("CSV saved to", csv_path)

print("\n===== Sorted Finished Results =====")
finished = [r for r in rows if r["status"] == "finished"]
finished.sort(key=lambda x: float(x["best_f1"]), reverse=True)

for r in finished:
    print(f'{r["group"]}\t{r["best_f1"]}\tepoch {r["best_epoch"]}\t{r["file"]}')
PY

cat "$SUMMARY_TXT"

echo "=========================================="
echo "Summary saved to:"
echo "$SUMMARY_TXT"
echo "$SUMMARY_CSV"
echo "All 24G disentanglement probe and ablation experiments finished!"
echo "=========================================="
