#!/bin/bash

set -e
set -o pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p Log/IEMOCAP/LargeScale
mkdir -p Log/MELD/LargeScale
mkdir -p Log/Summary

run_cmd () {
  LOG_FILE=$1
  shift

  if [ -f "$LOG_FILE" ] && grep -q "Best test f1" "$LOG_FILE"; then
    echo "Skip finished job: $LOG_FILE"
    return 0
  fi

  echo "=========================================="
  echo "Start job: $LOG_FILE"
  echo "Time: $(date)"
  echo "=========================================="

  "$@" 2>&1 | tee "$LOG_FILE"

  echo "=========================================="
  echo "Finished job: $LOG_FILE"
  echo "Time: $(date)"
  echo "=========================================="
}

# ============================================================
# IEMOCAP: 10 seeds, 150 epochs
# ============================================================

# IEMOCAP_SEEDS=(2025 2026 2027 1234 42 3037 6666 8888)

# for SEED in "${IEMOCAP_SEEDS[@]}"
# do
#   LOG_FILE="Log/IEMOCAP/LargeScale/IEMOCAP_final_seed${SEED}_epoch150.put"

#   run_cmd "$LOG_FILE" \
#     python Train/TrainMultiEMO.py \
#       --dataset 'IEMOCAP' \
#       --batch_size 64 \
#       --num_layers 6 \
#       --num_epochs 150 \
#       --SWFC_loss_param 0.4 \
#       --HGR_loss_param 0.2 \
#       --CE_loss_param 0.4 \
#       --aux_loss_param 0.2 \
#       --cmcl_loss_param 0 \
#       --cmcl_temp_param 0.5 \
#       --sample_weight_param 1.1 \
#       --temp_param 0.8 \
#       --focus_param 2.4 \
#       --dropout_rate 0.1 \
#       --dropout_rec 0.1 \
#       --seed ${SEED}
# done

# ============================================================
# MELD: 4 configs x 5 seeds = 20 experiments
# ============================================================

MELD_SEEDS=(2023 2024 2025 2026 2027)

run_meld_config () {
  CONFIG_NAME=$1
  CMCL_W=$2
  CMCL_T=$3
  LS=$4

  for SEED in "${MELD_SEEDS[@]}"
  do
    LOG_FILE="Log/MELD/LargeScale/MELD_${CONFIG_NAME}_seed${SEED}_epoch25.put"

    run_cmd "$LOG_FILE" \
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
        --meld_label_smoothing ${LS} \
        --sample_weight_param 1.2 \
        --temp_param 1.4 \
        --focus_param 2.0 \
        --dropout_rate 0 \
        --dropout_rec 0 \
        --seed ${SEED}
  done
}

run_meld_config "noCMCL_ls005"      0     0.5 0.05
run_meld_config "cmcl0025_temp05"   0.025 0.5 0.05
run_meld_config "cmcl003_temp05"    0.03  0.5 0.05
run_meld_config "cmcl0035_temp05"   0.035 0.5 0.05

# ============================================================
# Collect summary
# ============================================================

echo "=========================================="
echo "Collecting Summary"
echo "=========================================="

SUMMARY_FILE="Log/Summary/large_scale_summary.txt"
CSV_FILE="Log/Summary/large_scale_summary.csv"

rm -f "$SUMMARY_FILE"
rm -f "$CSV_FILE"

echo "===== IEMOCAP Large Scale Results =====" >> "$SUMMARY_FILE"

for f in Log/IEMOCAP/LargeScale/*.put
do
  echo "$(basename "$f")" >> "$SUMMARY_FILE"
  grep "Best test f1" "$f" >> "$SUMMARY_FILE" || true
  echo "" >> "$SUMMARY_FILE"
done

echo "===== MELD Large Scale Results =====" >> "$SUMMARY_FILE"

for f in Log/MELD/LargeScale/*.put
do
  echo "$(basename "$f")" >> "$SUMMARY_FILE"
  grep "Best test f1" "$f" >> "$SUMMARY_FILE" || true
  echo "" >> "$SUMMARY_FILE"
done

python - <<'PY'
import os
import re
import glob
import csv
import math
from collections import defaultdict

rows = []

patterns = [
    ("IEMOCAP", "Log/IEMOCAP/LargeScale/*.put"),
    ("MELD", "Log/MELD/LargeScale/*.put"),
]

for dataset, pattern in patterns:
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        m = re.search(r"Best test f1:\s*([0-9.]+)\s*at epoch\s*([0-9]+)", text)
        if not m:
            continue

        best_f1 = float(m.group(1))
        epoch = int(m.group(2))

        seed_match = re.search(r"seed([0-9]+)", name)
        seed = seed_match.group(1) if seed_match else ""

        if dataset == "IEMOCAP":
            config = "final"
        else:
            config_match = re.search(r"MELD_(.*?)_seed", name)
            config = config_match.group(1) if config_match else ""

        rows.append({
            "file": name,
            "dataset": dataset,
            "config": config,
            "seed": seed,
            "best_f1": best_f1,
            "epoch": epoch,
        })

csv_path = "Log/Summary/large_scale_summary.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["file", "dataset", "config", "seed", "best_f1", "epoch"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("CSV saved to", csv_path)

groups = defaultdict(list)

for r in rows:
    key = (r["dataset"], r["config"])
    groups[key].append(r["best_f1"])

print("\n===== Mean ± Std =====")

for key, values in sorted(groups.items()):
    mean = sum(values) / len(values)

    if len(values) > 1:
        var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        std = math.sqrt(var)
    else:
        std = 0.0

    print(
        f"{key[0]} / {key[1]}: {mean:.4f} ± {std:.4f}  n={len(values)}"
    )
PY

cat "$SUMMARY_FILE"

echo "=========================================="
echo "Summary files:"
echo "$SUMMARY_FILE"
echo "$CSV_FILE"
echo "=========================================="

echo "All large-scale experiments finished!"
