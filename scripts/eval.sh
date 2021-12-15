#!/usr/bin/env bash
echo usage: 
echo scriptName.sh : run in normal mode
echo scriptName.sh debug : run in debug mode

# hardware
cudaID=$2

# debug mode
if [[ $# != 0 ]] && [[ $1 == "debug" ]]
then
    debug=true
else
    debug=false
fi

seed=1

# dataset
dataDir='bank77'
targetDomain="BANKING"
dataDir=mcid
targetDomain="MEDICAL"
dataDir=hint3
targetDomain='curekart,powerplay11,sofmattress'

# setting
shot=2

# model initialization
LMName=intent-bert-base-uncased
# LMName=joint-intent-bert-base-uncased-bank77
# LMName=joint-intent-bert-base-uncased-mcid
# LMName=joint-intent-bert-base-uncased-hint3

# modify arguments if it's debug mode
RED='\033[0;31m'
GRN='\033[0;32m'
NC='\033[0m' # No Color
if $debug
then
    echo -e "Run in ${RED} debug ${NC} mode."
    epochs=1
else
    echo -e "Run in ${GRN} normal ${NC} mode."
fi

echo "Start Experiment ..."
logFolder=./log/
mkdir -p ${logFolder}
logFile=${logFolder}/eval_${dataDir}_${way}way_${shot}shot_LMName${LMName}.log
if $debug
then
	logFlie=${logFolder}/logDebug.log
fi

export CUDA_VISIBLE_DEVICES=${cudaID}
python eval.py \
	--seed ${seed} \
    	--targetDomain ${targetDomain} \
    	--dataDir ${dataDir} \
    	--shot ${shot}  \
    	--LMName ${LMName} \
    	| tee "${logFile}"
echo "Experiment finished."
