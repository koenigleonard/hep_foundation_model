#!/usr/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=00:03:00
#SBATCH --job-name=prob_cosine
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --account=rwth0934
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -p c23g
#SBATCH --array=0-3

# activate conda
source ~/miniforge/etc/profile.d/conda.sh
conda activate torchgpu

mkdir -p logs

python util/gitversion.py

DATASET="JETCLASS"
FOLDER="${DATASET}_600000_constant"

TAG="sampled"
N_JETS=50000
BATCH_SIZE=50
NUM_CONST=100

classes=("TTBar" "QCD")

OUTPUTPATH="output/plot_data/probs_best_${FOLDER}"
mkdir -p "$OUTPUTPATH"

# map array index → (i,j)
i=$((SLURM_ARRAY_TASK_ID / 2))
j=$((SLURM_ARRAY_TASK_ID % 2))

MODELFILE="/hpcwork/rwth0934/hep_foundation_model/training/${FOLDER}/checkpoints/${classes[$i]}_best.pt"
INPUTFILE="/hpcwork/rwth0934/hep_foundation_model/sampled_jets/${classes[$j]}_${FOLDER}_50000_topk.h5"
OUTPUTFILE="${OUTPUTPATH}/${classes[$i]}_${classes[$j]}_${TAG}.csv"

python compute_probabilities.py \
    --model_path "$MODELFILE" \
    --data_path "$INPUTFILE" \
    --output_file "$OUTPUTFILE" \
    --n_jets $N_JETS \
    --batch_size $BATCH_SIZE \
    --num_const $NUM_CONST \
    --input_key "sampled_jets" \
    --h5