#!/usr/bin/env bash

FREE_GPUS=$1

IFS=,; for gpu_id in ${FREE_GPUS};
do
    echo GPU_RESERVED_${gpu_id}
    declare -x -g GPU_RESERVED_${gpu_id}=false
done
