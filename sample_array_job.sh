#!/usr/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --job-name=sample_epochs
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --account=rwth0934
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -p c23g
#SBATCH --array=0-9

source ~/miniforge/etc/profile.d/conda.sh
conda activate torchgpu

module purge
module load CUDA
module load intel

mkdir -p logs

DATASET="JETCLASS"
FOLDER="${DATASET}_600000_warmup_cosine"

TRAINING_PATH="/hpcwork/rwth0934/hep_foundation_model/training/${FOLDER}"
OUTPUT_PATH="output/evaluation/${FOLDER}"

EPOCHS=(0 2 4 6 8 10 12 14 16 18)

EPOCH=${EPOCHS[$SLURM_ARRAY_TASK_ID]}

python sample_all_epochs.py \
    --training_folder "$TRAINING_PATH" \
    --output "$OUTPUT_PATH" \
    --epoch $EPOCH