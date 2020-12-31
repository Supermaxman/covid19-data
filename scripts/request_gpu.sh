#!/usr/bin/env bash

MEMORY_THRESHOLD=200
NUM_GPUS_REQUESTED=$1

GPU_MEMORY=`nvidia-smi --query-gpu=index,memory.used --format=csv`
i=0
n=0
OUTPUTS=
while IFS=, read -r gpu_id line2
do
    if [[ ${i} -gt 0 ]]; then
        j=0
        for gpu_mem in ${line2}
        do
            if [[ ${j} -eq 0 ]]; then
                if [[ ${gpu_mem} -lt ${MEMORY_THRESHOLD} ]]; then
                    v=GPU_RESERVED_${gpu_id}
                    if [[ ${!v} = true ]]; then
                        echo RESERVED
                    else
                        echo ${gpu_id}
                        if [[ ${n} -gt 0 ]]; then
                            OUTPUTS=${OUTPUTS},
                        fi
                        OUTPUTS=${OUTPUTS}${gpu_id}
                        declare -x GPU_RESERVED_${gpu_id}=true
                        n=$((n+1))
                    fi

                fi
            fi
            j=$((j+1))
        done
    fi
    i=$((i+1))
    if [[ ${n} -eq ${NUM_GPUS_REQUESTED} ]]; then
        break
    fi
done <<< "${GPU_MEMORY}"


if [[ ${n} -eq ${NUM_GPUS_REQUESTED} ]]; then
    echo ${OUTPUTS}
else
    echo -1
fi

