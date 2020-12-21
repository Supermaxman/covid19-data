#!/usr/bin/env bash

# run names
RUN_ID=7

# collection
DATASET=covid-lies

# major hyper-parameters for system
QA_PRE_MODEL_NAME=digitalepidemiologylab/covid-twitter-bert-v2
#export QA_PRE_MODEL_NAME=nboost/pt-biobert-base-msmarco
QA_THRESHOLD=0.1

QA_TRAIN_GPUS=7
QA_EVAL_GPUS=7
RETRIEVAL_EVAL_GPUS=4,5,6,7

NUM_QA_SPLITS=5
# qa flags
# QA fine-tune qaing model using training set
TRAIN_QA=false
# QA run qa using trained model on validation set
RUN_QA=false

RUN_RETRIEVAL=false
# QA run evaluation script on validation set
EVAL_QA=false

EVAL_RETRIEVAL=true

DATASET_PATH=data
COLLECTION_PATH=${DATASET_PATH}/downloaded_tweets_labeled.jsonl

ARTIFACTS_PATH=artifacts/${DATASET}
QA_RUN_NAME=HLTRI_COVID_LIES_QA_${RUN_ID}
QA_RUN_PATH=${ARTIFACTS_PATH}/${QA_RUN_NAME}
QA_RUN_FILE_PATH=${QA_RUN_PATH}/${QA_RUN_NAME}.run
QA_EVAL_FILE_PATH=${QA_RUN_PATH}/${QA_RUN_NAME}.eval
QA_SPLIT_FILES=""


RETRIEVAL_RUN_NAME=HLTRI_COVID_LIES_RETRIEVAL_${RUN_ID}
RETRIEVAL_RUN_PATH=${ARTIFACTS_PATH}/${RETRIEVAL_RUN_NAME}
RETRIEVAL_RUN_FILE_PATH=${RETRIEVAL_RUN_PATH}/${RETRIEVAL_RUN_NAME}.run
RETRIEVAL_EVAL_FILE_PATH=${RETRIEVAL_RUN_PATH}/${RETRIEVAL_RUN_NAME}.eval
RETRIEVAL_SPLIT_FILES=""
# python qa/create_split.py -i ${COLLECTION_PATH} -o ${DATASET_PATH}

for (( SPLIT=1; SPLIT<=${NUM_QA_SPLITS}; SPLIT++ )) do
    QA_SPLIT_RUN_NAME=HLTRI_COVID_LIES_QA_SPLIT_${SPLIT}_${RUN_ID}
    QA_SPLIT_RUN_MODEL_NAME=HLTRI_COVID_LIES_QA_SPLIT_${SPLIT}_${RUN_ID}
    QA_SPLIT_MODEL_NAME=qa-${DATASET}-${QA_SPLIT_RUN_MODEL_NAME}
    QA_SPLIT_PATH=${ARTIFACTS_PATH}/${QA_SPLIT_RUN_NAME}
    QA_SPLIT_FILE_PATH=${QA_SPLIT_PATH}/${QA_SPLIT_RUN_NAME}.qa
    RETRIEVAL_SPLIT_RUN_NAME=HLTRI_COVID_LIES_RETRIEVAL_${SPLIT}_${RUN_ID}
    RETRIEVAL_SPLIT_PATH=${ARTIFACTS_PATH}/${RETRIEVAL_SPLIT_RUN_NAME}
    RETRIEVAL_SPLIT_FILE_PATH=${RETRIEVAL_SPLIT_PATH}/${RETRIEVAL_SPLIT_RUN_NAME}.re

    if [[ ${TRAIN_QA} = true ]]; then
        echo "Training split ${SPLIT} qa model..."
        python qa/qa_train.py \
          --split_path ${DATASET_PATH}/split_${SPLIT}.json \
          --pre_model_name ${QA_PRE_MODEL_NAME} \
          --model_name ${QA_SPLIT_MODEL_NAME} \
          --max_seq_len 128 \
          --batch_size 8 \
          --learning_rate 5e-5 \
          --epochs 20 \
          --gpus ${QA_TRAIN_GPUS}
    fi

    if [[ ${RUN_QA} = true ]]; then
        echo "Running split ${SPLIT} qa..."
        python qa/qa_predict.py \
          --split_path ${DATASET_PATH}/split_${SPLIT}.json \
          --pre_model_name ${QA_PRE_MODEL_NAME} \
          --model_name ${QA_SPLIT_MODEL_NAME} \
          --output_path ${QA_SPLIT_PATH} \
          --max_seq_len 128 \
          --batch_size 8 \
          --learning_rate 5e-5 \
          --epochs 20 \
          --load_trained_model \
          --gpus ${QA_EVAL_GPUS} \
        ; \
        python qa/format_qa_predictions.py \
          --input_path ${QA_SPLIT_PATH} \
          --output_path ${QA_SPLIT_FILE_PATH}
    fi
    if [[ ${RUN_RETRIEVAL} = true ]]; then
        echo "Running split ${SPLIT} retrieval..."
        python qa/qa_predict.py \
          --split_path ${DATASET_PATH}/split_${SPLIT}.json \
          --pre_model_name ${QA_PRE_MODEL_NAME} \
          --model_name ${QA_SPLIT_MODEL_NAME} \
          --output_path ${RETRIEVAL_SPLIT_PATH} \
          --max_seq_len 128 \
          --batch_size 16 \
          --learning_rate 5e-5 \
          --epochs 20 \
          --load_trained_model \
          --gpus ${RETRIEVAL_EVAL_GPUS} \
          --mode retrieval \
          --misconceptions_path ${DATASET_PATH}/misconceptions.json  \
        ; \
        python qa/format_qa_predictions.py \
          --input_path ${RETRIEVAL_SPLIT_PATH} \
          --output_path ${RETRIEVAL_SPLIT_FILE_PATH}
    fi

    QA_SPLIT_FILES="${QA_SPLIT_FILES},${QA_SPLIT_FILE_PATH}"
    RETRIEVAL_SPLIT_FILES="${RETRIEVAL_SPLIT_FILES},${RETRIEVAL_SPLIT_FILE_PATH}"
done

if [[ ${EVAL_QA} = true ]]; then
    echo "Evaluating qa model..."
    mkdir -p ${QA_RUN_PATH}
    python qa/format_qa_eval.py \
      --input_path ${QA_SPLIT_FILES} \
      --output_path ${QA_RUN_FILE_PATH} \
      --threshold ${QA_THRESHOLD}

    python qa/qa_eval.py \
      --label_path ${COLLECTION_PATH} \
      --run_path ${QA_RUN_FILE_PATH} \
      > ${QA_EVAL_FILE_PATH} \
      ; \
      cat ${QA_EVAL_FILE_PATH}
fi


if [[ ${EVAL_RETRIEVAL} = true ]]; then
    echo "Evaluating qa model..."
    mkdir -p ${RETRIEVAL_RUN_PATH}
    python qa/format_retrieval_eval.py \
      --input_path ${RETRIEVAL_SPLIT_FILES} \
      --output_path ${RETRIEVAL_RUN_FILE_PATH}

    python qa/retrieval_eval.py \
      --label_path ${COLLECTION_PATH} \
      --run_path ${RETRIEVAL_RUN_FILE_PATH} \
      > ${RETRIEVAL_EVAL_FILE_PATH} \
      ; \
      cat ${RETRIEVAL_EVAL_FILE_PATH}
fi
