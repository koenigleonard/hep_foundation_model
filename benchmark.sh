#---- activate conda
source ~/miniforge/etc/profile.d/conda.sh
conda activate torchgpu

INPUTFILE="/hpcwork/rwth0934/hep_foundation_model/preprocessed_data/Top_train_discrete_pT_eta_phi.h5"
#INPUTFILE="processed_data/TTBar_5000_processed_train.h5"
NAME="BENCHMARK"
NUM_CONST=50
N_JETS=5000
N_JETS_VAL=2000
BATCH_SIZE=100

#print version of repo:
python util/gitversion.py

python benchmark.py --data_path "$INPUTFILE" --output_path output/ --name "$NAME" --num_const $NUM_CONST --n_jets $N_JETS --n_jets_val $N_JETS_VAL --input_key df --batch_size $BATCH_SIZE