#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --job-name=sample_jets
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --account=rwth0934
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -p c23g
#SBATCH --array=0-1   # adjust if more classes

# activate conda
source ~/miniforge/etc/profile.d/conda.sh
conda activate torchgpu

export HDF5_USE_FILE_LOCKING=FALSE

module purge
module load CUDA
module load intel

mkdir -p logs

# parameters
DATASET="JETCLASS"
FOLDER="${DATASET}_10000000_cosine_restarts_128"
N_JETS=50000
BATCH_SIZE=500
MAX_LENGTH=128
TOPK=5000

# classes to sample
classes=("TTBar" "QCD")

# select class based on array index
CLASS=${classes[$SLURM_ARRAY_TASK_ID]}

# paths
INPUTFILE="/hpcwork/rwth0934/hep_foundation_model/training/${FOLDER}/checkpoints/${CLASS}_best.pt"
OUTPUTDIR="/hpcwork/rwth0934/hep_foundation_model/sampled_jets"
OUTPUTFILE="${OUTPUTDIR}/${CLASS}_${FOLDER}_${N_JETS}_topk.h5"

mkdir -p "$OUTPUTDIR"

# print git version
python util/gitversion.py

# run sampling
python sample.py \
    --model_path "$INPUTFILE" \
    --output_file "$OUTPUTFILE" \
    --n_jets $N_JETS \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --topk $TOPK