#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --ntasks=1              
#SBATCH --time=01:00:00         
#SBATCH --job-name=preprocess_job
#SBATCH --output=logs/%x_%j.out
#SBATCH --account=rwth0934  # Replace with your project-id or delete the line

### Program Code
#---- activate conda
source ~/miniforge/etc/profile.d/conda.sh
conda activate hep_foundation_model

#---- create log dir
mkdir -p logs

INPUTFILE="/hpcwork/rwth0934/hep_foundation_model_data/JetClass/TTBar_train.h5"
OUTPUTFILE="/hpcwork/rwth0934/hep_foundation_model_data/preprocessed_data/TTBar_train_processed.h5"

python preprocess.py --input_file "$INPUTFILE" --output_file "$OUTPUTFILE" --all