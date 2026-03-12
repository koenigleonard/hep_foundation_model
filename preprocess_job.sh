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
conda activate torchgpu

#---- create log dir
mkdir -p logs

INPUTFILE="/hpcwork/rwth0934/hep_foundation_model/JetClass/ZJetsToNuNu_test.h5"
OUTPUTFILE="/hpcwork/rwth0934/hep_foundation_model/preprocessed_data/QCD_test_processed.h5"
#OUTPUTFILE=processed_data/TTBar_600000_val.h5
N_JETS=600000

python preprocess.py --input_file "$INPUTFILE" \
                     --output_file "$OUTPUTFILE" \
                     --all \
                     --pt_min -0.7633162140846252 --pt_max 6.748834133148193