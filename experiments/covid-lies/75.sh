#!/usr/bin/env bash

filename=$(basename -- "$0")
# run names
RUN_ID=${filename::-3}
RUN_NAME=HLTRI_COVID_LIES_STANCE

# collection
DATASET=covid-lies
NUM_STANCE_SPLITS=5
CREATE_SPLIT=false
SPLIT_TYPE=normal_unique

# major hyper-parameters for system
STANCE_PRE_MODEL_NAME=digitalepidemiologylab/covid-twitter-bert-v2
STANCE_THRESHOLD=0.2

STANCE_BATCH_SIZE=8
STANCE_MAX_SEQ_LEN=128

STANCE_NUM_GPUS=1
TRAIN_STANCE=true
RUN_STANCE=true
EVAL_STANCE=true

export TOKENIZERS_PARALLELISM=true

echo "Starting experiment ${RUN_NAME}_${RUN_ID}"
echo "Reserving ${STANCE_NUM_GPUS} GPU(s)..."
STANCE_GPUS=`python gpu/request_gpus.py -r ${STANCE_NUM_GPUS}`
if [[ ${STANCE_GPUS} -eq -1 ]]; then
    echo "Unable to reserve ${STANCE_NUM_GPUS} GPU(s), exiting."
    exit -1
fi
echo "Reserved ${STANCE_NUM_GPUS} GPUs: ${STANCE_GPUS}"
STANCE_TRAIN_GPUS=${STANCE_GPUS}
STANCE_EVAL_GPUS=${STANCE_GPUS}

DATASET_PATH=data
ARTIFACTS_PATH=artifacts/${DATASET}
STANCE_SPLIT_FILES=""

# trap ctrl+c to free GPUs
handler()
{
    echo "Experiment aborted."
    echo "Freeing ${STANCE_NUM_GPUS} GPUs: ${STANCE_GPUS}"
    python gpu/free_gpus.py -i ${STANCE_GPUS}
    exit -1
}
trap handler SIGINT

if [[ ${CREATE_SPLIT} = true ]]; then
    echo "Creating ${SPLIT_TYPE} splits..."
    python stance/create_split.py -i ${DATASET_PATH}/downloaded_tweets_labeled.jsonl -o ${DATASET_PATH} -t ${SPLIT_TYPE}
fi

for (( SPLIT=1; SPLIT<=${NUM_STANCE_SPLITS}; SPLIT++ )) do
    if [[ ${TRAIN_STANCE} = true ]]; then
        echo "Training split ${SPLIT} stance model..."
        python stance/stance_train.py \
          --model_type lm-gcn \
          --create_edge_features \
          --graph_names semantic,emotion,lexical \
          --gcn_size 64 \
          --gcn_depth 6 \
          --gcn_type attention \
          --misconception_info_path ${DATASET_PATH}/misconceptions_extra.json \
          --split_path ${DATASET_PATH}/${SPLIT_TYPE}_split_${SPLIT}.json \
          --pre_model_name ${STANCE_PRE_MODEL_NAME} \
          --model_name stance-${DATASET}-${RUN_NAME}_SPLIT_${SPLIT}_${RUN_ID} \
          --max_seq_len ${STANCE_MAX_SEQ_LEN} \
          --batch_size ${STANCE_BATCH_SIZE} \
          --learning_rate 5e-4 \
          --epochs 10 \
          --fine_tune \
          --gpus ${STANCE_TRAIN_GPUS}
    fi

    if [[ ${RUN_STANCE} = true ]]; then
        echo "Running split ${SPLIT} stance..."
        python stance/stance_predict.py \
          --model_type lm-gcn \
          --create_edge_features \
          --graph_names semantic,emotion,lexical \
          --gcn_size 64 \
          --gcn_depth 6 \
          --gcn_type attention \
          --misconception_info_path ${DATASET_PATH}/misconceptions_extra.json \
          --split_path ${DATASET_PATH}/${SPLIT_TYPE}_split_${SPLIT}.json \
          --pre_model_name ${STANCE_PRE_MODEL_NAME} \
          --model_name stance-${DATASET}-${RUN_NAME}_SPLIT_${SPLIT}_${RUN_ID} \
          --output_path ${ARTIFACTS_PATH}/${RUN_NAME}_SPLIT_${SPLIT}_${RUN_ID} \
          --max_seq_len ${STANCE_MAX_SEQ_LEN} \
          --batch_size ${STANCE_BATCH_SIZE} \
          --load_trained_model \
          --gpus ${STANCE_EVAL_GPUS} \
        ; \
        python stance/format_stance_predictions.py \
          --input_path ${ARTIFACTS_PATH}/${RUN_NAME}_SPLIT_${SPLIT}_${RUN_ID} \
          --output_path ${ARTIFACTS_PATH}/${RUN_NAME}_SPLIT_${SPLIT}_${RUN_ID}/predictions.stance
    fi

    STANCE_SPLIT_FILES="${STANCE_SPLIT_FILES},${ARTIFACTS_PATH}/${RUN_NAME}_SPLIT_${SPLIT}_${RUN_ID}/predictions.stance"
done

echo "Freeing ${STANCE_NUM_GPUS} GPUs: ${STANCE_GPUS}"
python gpu/free_gpus.py -i ${STANCE_GPUS}

if [[ ${EVAL_STANCE} = true ]]; then
    echo "Evaluating stance model..."
    mkdir -p ${ARTIFACTS_PATH}/${RUN_NAME}_${RUN_ID}
    python stance/format_stance_eval.py \
      --input_path ${STANCE_SPLIT_FILES} \
      --output_path ${ARTIFACTS_PATH}/${RUN_NAME}_${RUN_ID}/all.run \
      --threshold ${STANCE_THRESHOLD}

    python stance/stance_eval.py \
      --label_path ${DATASET_PATH}/downloaded_tweets_labeled.jsonl \
      --run_path ${ARTIFACTS_PATH}/${RUN_NAME}_${RUN_ID}/all.run \
      > ${ARTIFACTS_PATH}/${RUN_NAME}_${RUN_ID}/eval.txt \
      ; \
      cat ${ARTIFACTS_PATH}/${RUN_NAME}_${RUN_ID}/eval.txt
fi


