#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --ntasks=1              
#SBATCH --time=02:00:00         
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

#---- create log dir
mkdir -p logs

INPUTFILE="/hpcwork/rwth0934/hep_foundation_model/preprocessed_data/Top_train_discrete_pT_eta_phi.h5"
#INPUTFILE="processed_data/TTBar_5000_processed_train.h5"
NAME="TOP_600000"

#print version of repo:
python util/gitversion.py

python train.py --data_path "$INPUTFILE" --output_path output/ --name "$NAME" --num_const 50 --num_epochs 20 --n_jets 600000 --n_jets_val 200000 --input_key df