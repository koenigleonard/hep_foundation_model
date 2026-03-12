#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --ntasks=1              
#SBATCH --time=01:00:00         
#SBATCH --job-name=compute_metrics
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

export HDF5_USE_FILE_LOCKING=FALSE

module purge
module load CUDA
module load intel

#---- create log dir
mkdir -p logs

#INPUTFILE="/hpcwork/rwth0934/hep_foundation_model/sampled_jets/TTBar_600000_FACTORIZED_sampled_50000_topk.h5"
INPUTFILE=output/sampled_jets/test_jets.h5
OUTPUTFILE="output/plot_data_jetclass/metrics/TTBar_600000_FACTORIZED_sampled_50000_topk.h5"

N_JETS=50000
BATCH_SIZE=100
NUM_CONST=200
INPUTKEY="sampled_jets"

#print version of repo:
python util/gitversion.py

python compute_metrics.py --data_path "$INPUTFILE" \
                 --output_file "$OUTPUTFILE" \
                 --batch_size $BATCH_SIZE \
                 --num_const $NUM_CONST \
                 --pt_min -0.7633162140846252 --pt_max 6.748834133148193 \
                 --eta_min -0.8 --eta_max 0.8 \
                 --phi_min -0.8 --phi_max 0.8