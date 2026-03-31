#!/usr/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --job-name=eval_mixed
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --account=rwth0934
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -p c23g

source ~/miniforge/etc/profile.d/conda.sh
conda activate torchgpu

module purge
module load CUDA
module load intel

mkdir -p logs

python util/gitversion.py

# -----------------------------
# CONFIG
# -----------------------------
DATASET="JETCLASS"
FOLDER="${DATASET}_10000000_cosine_restarts_50"

TRAINING_PATH="/hpcwork/rwth0934/hep_foundation_model/training/${FOLDER}"
DATA_PATH="/hpcwork/rwth0934/hep_foundation_model"
OUTPUT_PATH="output/evaluation/${FOLDER}"

N_JETS=50000
BATCH_SIZE=50
NUM_CONST=100

# -----------------------------
# EPOCH SCAN
# -----------------------------
EPOCHS=(0 2 4 6 8 10 12 14 16 18 20 22 24 26 28)

# -----------------------------
# MODEL MODES
# -----------------------------
# Options: epoch, best, untrained

TTBAR_MODE="best"
QCD_MODE="epoch"

# -----------------------------
# EPOCH ASSIGNMENT
# -----------------------------
if [ "$TTBAR_MODE" = "epoch" ]; then
    TTBAR_EPOCH=${EPOCHS[$SLURM_ARRAY_TASK_ID]}
fi

if [ "$QCD_MODE" = "epoch" ]; then
    QCD_EPOCH=${EPOCHS[$SLURM_ARRAY_TASK_ID]}
fi

echo "TTBar mode: $TTBAR_MODE epoch: $TTBAR_EPOCH"
echo "QCD mode: $QCD_MODE epoch: $QCD_EPOCH"

# -----------------------------
# RUN
# -----------------------------
python evaluate_classifier.py \
    --training_folder "$TRAINING_PATH" \
    --data_folder "$DATA_PATH" \
    --output "$OUTPUT_PATH" \
    --ttbar_mode "$TTBAR_MODE" \
    --qcd_mode "$QCD_MODE" \
    ${TTBAR_EPOCH:+--ttbar_epoch $TTBAR_EPOCH} \
    ${QCD_EPOCH:+--qcd_epoch $QCD_EPOCH} \
    --n_jets $N_JETS \
    --batch_size $BATCH_SIZE \
    --num_const $NUM_CONST