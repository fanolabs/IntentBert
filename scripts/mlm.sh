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
# dataDir='mcid'
# dataDir='hint3'
targetDomain="BANKING"
# targetDomain='MEDICAL'
# targetDomain='curekart,powerplay11,sofmattress'

# setting
shot=2

# training
tensorboard=
saveModel=--saveModel
saveName=none

# model setting
# common
LMName=bert-base-uncased

# modify arguments if it's debug mode
RED='\033[0;31m'
GRN='\033[0;32m'
NC='\033[0m' # No Color
if $debug
then
    echo -e "Run in ${RED} debug ${NC} mode."
    # validationTaskNum=10
    # testTaskNum=10
    epochs=1
    # repeatNum=1
else
    echo -e "Run in ${GRN} normal ${NC} mode."
fi

echo "Start Experiment ..."
logFolder=./log/
mkdir -p ${logFolder}
logFile=${logFolder}/mlm_${sourceDomainName}_to_${targetDomainName}_${way}way_${shot}shot.log
if $debug
then
	logFlie=${logFolder}/logDebug.log
fi

export CUDA_VISIBLE_DEVICES=${cudaID}
python mlm.py \
	--seed ${seed} \
    	--targetDomain ${targetDomain} \
    	${tensorboard} \
    	--dataDir ${dataDir} \
    	--shot ${shot}  \
    	${saveModel} \
    	--LMName ${LMName} \
    	--saveName ${saveName} \
    	| tee "${logFile}"
echo "Experiment finished."
