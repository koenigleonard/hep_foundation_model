#!/bin/bash

# first argument specified in condor_submit_myAnalysis.condor at point arguments =
CONDOR_CLUSTER_ID=$1
# second argument specified in condor_submit_myAnalysis.condor at point arguments =
CONDOR_PROCESS_ID=$2

CONDOR_JOBID=${CONDOR_CLUSTER_ID}.${CONDOR_PROCESS_ID}

# since getenv is used in condor_submit script HOSTNAME is set back to its proper value here
HOSTNAME=`hostname -s`

# Shared input/output locations
INPUTFILE=${HOME}/Documents/master_thesis/hep_foundation_model/processed_data/debug_train.h5
OUTPUTDIR=${HOME}/Documents/master_thesis/hep_foundation_model/output
OUTPUTFILE=HToCC_train_processed.h5

WORKDIR=${HOME}/Documents/master_thesis/hep_foundation_model
PROGRAM=train.py

LOGFILE=logfile_${CONDOR_JOBID}-${HOSTNAME}.log

# while the job is executed on a machine within the cluster, the results will be stored temporarily in TMP_DATA_PATH
# usage of /user/scratch/ is mandatory for Physikzentrum
TMP_DATA_PATH=/user/scratch/koenig/koenig-condor-${CONDOR_JOBID}

# Disable core dumps
ulimit -c 0

# Clean + create scratch dir
rm -rf ${TMP_DATA_PATH}
mkdir -p ${TMP_DATA_PATH}

cd ${TMP_DATA_PATH}

echo "Job started on $(date)"        >> ${LOGFILE}
echo "Running on host ${HOSTNAME}"  >> ${LOGFILE}

# Copy script locally (fast execution)
cp ${WORKDIR}/${PROGRAM} ./

# ---- Activate Conda ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hep_foundation_model

echo "Conda environment:" >> ${LOGFILE}
which python             >> ${LOGFILE}

# ---- Run processing ----
echo "Starting Python job..." >> ${LOGFILE}

python ${PROGRAM} --data_path ${INPUTFILE} --output_path ${OUTPUTDIR} --num_const 50 2>&1 >> ${LOGFILE}

RES=$?
echo "Return code: ${RES}" >> ${LOGFILE}

# ---- Copy results back to shared storage ----
echo "Copying output back..." >> ${LOGFILE}

mkdir -p "${OUTPUTDIR}"
cp ${OUTPUTFILE} ${LOGFILE} ${OUTPUTDIR}/

# Cleanup scratch
rm -rf ${TMP_DATA_PATH}

exit ${RES}
