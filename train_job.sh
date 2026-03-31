#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --ntasks=1              
#SBATCH --time=00:45:00         
#SBATCH --job-name=top_restart
#SBATCH --output=logs/%x_%j.out
#SBATCH --account=thes2215  # Replace with your project-id or delete the line
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH -p c23g

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

CLASS=TTBar
SCHEDULER=constant
N_JETS=10000000
N_JETS_VAL=800000
NUM_CONST=50
NUM_EPOCHS=1
BATCH_SIZE=500
BATCH_SIZE_VAL=1000
LR=0.0001
GAMMA=0.9
DROPOUT=0.1

#print version of repo:
python util/gitversion.py

#FOLDER="${DATASET}_${N_JETS}_${SCHEDULER}_50"
FOLDER="${DATASET}_${N_JETS}_cosine_restarts_50"
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
                --linear_output \
                --restart_period 30 \
                --contin \
                --reset_scheduler \
                --checkpoint_name "${CLASS}_epoch_29.pt" \
                --new_lr 0.0001
