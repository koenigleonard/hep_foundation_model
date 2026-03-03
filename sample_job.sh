#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --ntasks=1              
#SBATCH --time=01:00:00         
#SBATCH --job-name=sample_qcd
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

INPUTFILE="/hpcwork/rwth0934/hep_foundation_model/checkpoints/checkpoints/QCD_600000_best.pt"
#INPUTFILE="processed_data/TTBar_5000_processed_train.h5"
#OUTPUTFILE="/hpcwork/rwth0934/hep_foundation_model/sampled_jets/QCD_600000_sampled_100000_topk.h5"
OUTPUTFILE=output/sampled_jets/test_jets.h5

#print version of repo:
python util/gitversion.py

python sample.py --model_path "$INPUTFILE" --output_file "$OUTPUTFILE" --n_jets 5000 --batch_size 100 --max_length 100 --topk 5000