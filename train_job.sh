#!/bin/bash

### Job Parameters 
#SBATCH --ntasks=1              
#SBATCH --time=04:00:00         
#SBATCH --job-name=const_leo_sl
#SBATCH --output=logs/%x_%j.out
#SBATCH --account=rwth0934  # Replace with your project-id or delete the line
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -p c23g
#SBATCH --array=0-1   # adjust if more classes

### Program Code
#---- activate conda
source ~/miniforge/etc/profile.d/conda.sh
conda activate torchgpu

#---- load modules
module purge
module load CUDA
module load intel

#---- create log dir
mkdir -p logs

DATASET=JETCLASS

SCHEDULER=const
N_JETS=600000
N_JETS_VAL=200000
NUM_CONST=50
NUM_EPOCHS=50
BATCH_SIZE=100
BATCH_SIZE_VAL=100
LR=0.0001
LR_MIN=1e-6
GAMMA=0.9
DROPOUT=0.0
WEIGHT_DECAY=0.00001

# # classes to train
classes=("TTBar" "QCD")

# select class based on array index
CLASS=${classes[$SLURM_ARRAY_TASK_ID]}
#CLASS=QCD

#print version of repo:
python util/gitversion.py

FOLDER="${DATASET}_${N_JETS}_${SCHEDULER}_weight_decay"
#FOLDER="${DATASET}_${N_JETS}_cosine_restarts_50_2"
NAME="${CLASS}"
INPUTFILE="/hpcwork/rwth0934/hep_foundation_model/preprocessed_data/${CLASS}_train_processed.h5"
OUTPUT_PATH="/hpcwork/rwth0934/hep_foundation_model/training/${FOLDER}"

python train.py --data_path "$INPUTFILE" \
                --output_path "$OUTPUT_PATH" \
                --name "$NAME" \
                --num_const $NUM_CONST \
                --num_epochs $NUM_EPOCHS \
                --n_jets $N_JETS \
                --n_jets_val $N_JETS_VAL \
                --batch_size $BATCH_SIZE \
                --batch_size_val $BATCH_SIZE_VAL \
                --lr $LR \
                --scheduler $SCHEDULER \
                --gamma $GAMMA \
                --dropout $DROPOUT \
                --weight_decay $WEIGHT_DECAY \
                --linear_output \
                --restart_period 36
