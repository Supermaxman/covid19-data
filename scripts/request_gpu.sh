#!/usr/bin/env bash

NUM_GPUS_REQUESTED=$1

AVAILABLE_GPUS=`bash scripts/check_gpus.sh`
n=0
OUTPUTS=
IFS=,; for gpu_id in ${AVAILABLE_GPUS};
do
    if [[ ${n} -gt 0 ]]; then
        OUTPUTS=${OUTPUTS},
    fi
    OUTPUTS=${OUTPUTS}${gpu_id}
    declare -x -g GPU_RESERVED_${gpu_id}=true
    n=$((n+1))
    if [[ ${n} -eq ${NUM_GPUS_REQUESTED} ]]; then
        break
    fi
done

if [[ ${n} -eq ${NUM_GPUS_REQUESTED} ]]; then
    echo ${OUTPUTS}
else
    bash scripts/free_gpus.sh ${OUTPUTS}
    echo -1
fi

