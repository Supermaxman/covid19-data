#!/usr/bin/env bash

# run names
RUN_ID=20
RUN_NAME=HLTRI_COVID_LIES_STANCE

# collection
DATASET=covid-lies
NUM_STANCE_SPLITS=5

# major hyper-parameters for system
STANCE_PRE_MODEL_NAME=digitalepidemiologylab/covid-twitter-bert-v2
STANCE_THRESHOLD=0.3

STANCE_BATCH_SIZE=8
STANCE_MAX_SEQ_LEN=128
STANCE_TRAIN_GPUS=1
STANCE_EVAL_GPUS=1


TRAIN_STANCE=false
RUN_STANCE=false
EVAL_STANCE=true

DATASET_PATH=data
ARTIFACTS_PATH=artifacts/${DATASET}
STANCE_SPLIT_FILES=""

# python stance/create_split.py -i ${COLLECTION_PATH} -o ${DATASET_PATH}

for (( SPLIT=1; SPLIT<=${NUM_STANCE_SPLITS}; SPLIT++ )) do
    if [[ ${TRAIN_STANCE} = true ]]; then
        echo "Training split ${SPLIT} stance model..."
        python stance/stance_train.py \
          --split_path ${DATASET_PATH}/split_${SPLIT}.json \
          --pre_model_name ${STANCE_PRE_MODEL_NAME} \
          --model_name stance-${DATASET}-${RUN_NAME}_SPLIT_${SPLIT}_${RUN_ID} \
          --max_seq_len ${STANCE_MAX_SEQ_LEN} \
          --batch_size ${STANCE_BATCH_SIZE} \
          --learning_rate 5e-5 \
          --epochs 20 \
          --fine_tune \
          --gpus ${STANCE_TRAIN_GPUS}
    fi

    if [[ ${RUN_STANCE} = true ]]; then
        echo "Running split ${SPLIT} stance..."
        python stance/stance_predict.py \
          --split_path ${DATASET_PATH}/split_${SPLIT}.json \
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


