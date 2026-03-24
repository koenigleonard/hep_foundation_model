#!/usr/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --job-name=eval_all_tags
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
FOLDER="${DATASET}_600000_cosine_restarts_50_untrained"

TRAINING_PATH="/hpcwork/rwth0934/hep_foundation_model/training/${FOLDER}"
DATA_PATH="/hpcwork/rwth0934/hep_foundation_model"
OUTPUT_PATH="output/evaluation/${FOLDER}"

N_JETS=50000
BATCH_SIZE=500
NUM_CONST=100

# -----------------------------
# MODES
# -----------------------------
EPOCHS=(0 1)

USE_BEST=0
USE_UNTRAINED=0

# -----------------------------
# MODE SELECTION
# -----------------------------
if [ "$USE_BEST" -eq 1 ]; then
    MODE="best"
elif [ "$USE_UNTRAINED" -eq 1 ]; then
    MODE="untrained"
else
    MODE="epoch"
    EPOCH=${EPOCHS[$SLURM_ARRAY_TASK_ID]}
fi

echo "Running mode: $MODE"

# -----------------------------
# RUN
# -----------------------------
if [ "$MODE" = "best" ]; then

python evaluate_classifier.py \
    --training_folder "$TRAINING_PATH" \
    --data_folder "$DATA_PATH" \
    --output "$OUTPUT_PATH" \
    --best \
    --n_jets $N_JETS \
    --batch_size $BATCH_SIZE \
    --num_const $NUM_CONST

elif [ "$MODE" = "untrained" ]; then

python evaluate_classifier.py \
    --training_folder "$TRAINING_PATH" \
    --data_folder "$DATA_PATH" \
    --output "$OUTPUT_PATH" \
    --untrained \
    --n_jets $N_JETS \
    --batch_size $BATCH_SIZE \
    --num_const $NUM_CONST

else

python evaluate_classifier.py \
    --training_folder "$TRAINING_PATH" \
    --data_folder "$DATA_PATH" \
    --output "$OUTPUT_PATH" \
    --epoch $EPOCH \
    --n_jets $N_JETS \
    --batch_size $BATCH_SIZE \
    --num_const $NUM_CONST

fi