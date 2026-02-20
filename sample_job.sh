#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --ntasks=1              
#SBATCH --time=01:00:00         
#SBATCH --job-name=sample_job
#SBATCH --output=logs/%x_%j.out
#SBATCH --account=rwth0934  # Replace with your project-id or delete the line
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --exclude=n23g0001
#SBATCH -p c23g

### Program Code
#---- activate conda
source ~/miniforge/etc/profile.d/conda.sh
conda activate torchgpu

#---- create log dir
mkdir -p logs

INPUTFILE="/home/eu732103/master_thesis/hep_foundation_model/output/checkpoints/TTBar_600000_best.pt"
#INPUTFILE="processed_data/TTBar_5000_processed_train.h5"
OUTPUTFILE="/hpcwork/rwth0934/hep_foundation_model/sampled_jets"

#print version of repo:
python util/gitversion.py

python train.py --model_path "$INPUTFILE" --output_file "$OUTPUTFILE" --n_jets 100000 --batch_size 100 --max_length 100