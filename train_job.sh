#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --ntasks=1              
#SBATCH --time=04:00:00         
#SBATCH --job-name=train_top
#SBATCH --output=logs/%x_%j.out
#SBATCH --account=rwth0934  # Replace with your project-id or delete the line
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=n23g0001
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

INPUTFILE="/hpcwork/rwth0934/hep_foundation_model/preprocessed_data/Top_train_discrete_pT_eta_phi.h5"
#INPUTFILE="processed_data/TTBar_5000_processed_train.h5"
OUTPUT_PATH="/hpcwork/rwth0934/hep_foundation_model/checkpoints/"
NAME="TOP_600000"
N_JETS=600000
N_JETS_VAL=200000
NUM_CONST=50
NUM_EPOCHS=50
BATCH_SIZE=100

#print version of repo:
python util/gitversion.py

python train.py --data_path "$INPUTFILE" \
                --output_path "$OUTPUT_PATH" \
                --name "$NAME" \
                --num_const $NUM_CONST \
                --num_epochs $NUM_EPOCHS \
                --n_jets $N_JETS \
                --n_jets_val $N_JETS_VAL \
                --batch_size $BATCH_SIZE \
                --input_key df