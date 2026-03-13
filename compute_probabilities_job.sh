#!/usr/bin/env bash

### Job Parameters 
#SBATCH --ntasks=1              
#SBATCH --time=01:00:00         
#SBATCH --job-name=prob_job
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

#print version of repo:h/hep_foundation_model/outp
python util/gitversion.py

DATASET="JETCLASS"

TAG="val" #which dataset should be used
N_JETS=50000 #number of jets taken out of each test set
BATCH_SIZE=50
NUM_CONST=128

OUTPUTMODE="LINEAR"
#denots which epoch was the best in the respective trainings run 
epochs=(
    "8"
    "5"
)

classes=(
    "TTBar"
    "QCD"
)

for i in "${!epochs[@]}"; do
    for j in "${!classes[@]}"; do
        MODELFILE="/hpcwork/rwth0934/hep_foundation_model/checkpoints/checkpoints/${DATASET}_${classes[$i]}_600000_${OUTPUTMODE}_epoch_${epochs[$i]}.pt"
        INPUTFILE="/hpcwork/rwth0934/hep_foundation_model/preprocessed_data/${classes[$j]}_${TAG}_processed.h5"
        OUTPUTFILE="output/plot_data_jetclass/probs_best_epoch/${classes[$i]}_${OUTPUTMODE}_${classes[$j]}_${TAG}.csv"

        # echo "$MODELFILE"
        # echo "$INPUTFILE"
        # echo "$OUTPUTFILE"

        python compute_probabilities.py --model_path "$MODELFILE" \
                                        --data_path "$INPUTFILE" \
                                        --output_file "$OUTPUTFILE" \
                                        --n_jets $N_JETS \
                                        --batch_size $BATCH_SIZE \
                                        --num_const $NUM_CONST
    done
done

OUTPUTMODE="FACTORIZED"

epochs=(
    "16"
    "12"
)

classes=(
    "TTBar"
    "QCD"
)

for i in "${!epochs[@]}"; do
    for j in "${!classes[@]}"; do
        MODELFILE="/hpcwork/rwth0934/hep_foundation_model/checkpoints/checkpoints/${DATASET}_${classes[$i]}_600000_${OUTPUTMODE}_epoch_${epochs[$i]}.pt"
        INPUTFILE="/hpcwork/rwth0934/hep_foundation_model/preprocessed_data/${classes[$j]}_${TAG}_processed.h5"
        OUTPUTFILE="output/plot_data_jetclass/probs_best_epoch/${classes[$i]}_${OUTPUTMODE}_${classes[$j]}_${TAG}.csv"

        # echo "$MODELFILE"
        # echo "$INPUTFILE"
        # echo "$OUTPUTFILE"

        python compute_probabilities.py --model_path "$MODELFILE" \
                                        --data_path "$INPUTFILE" \
                                        --output_file "$OUTPUTFILE" \
                                        --n_jets $N_JETS \
                                        --batch_size $BATCH_SIZE \
                                        --num_const $NUM_CONST
    done
done