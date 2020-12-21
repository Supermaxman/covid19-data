#!/usr/bin/env bash

# run names
RUN_ID=7

# collection
DATASET=covid-lies

# major hyper-parameters for system
QA_PRE_MODEL_NAME=digitalepidemiologylab/covid-twitter-bert-v2
#export QA_PRE_MODEL_NAME=nboost/pt-biobert-base-msmarco
QA_THRESHOLD=0.1

GPUS=7
NUM_QA_SPLITS=5
# qa flags
# QA fine-tune qaing model using training set
TRAIN_QA=false
# QA run qa using trained model on validation set
RUN_QA=false
# QA run evaluation script on validation set
EVAL_QA=true

DATASET_PATH=data
COLLECTION_PATH=${DATASET_PATH}/downloaded_tweets_labeled.jsonl

ARTIFACTS_PATH=artifacts/${DATASET}
QA_RUN_NAME=HLTRI_COVID_LIES_QA_${RUN_ID}
QA_RUN_PATH=${ARTIFACTS_PATH}/${QA_RUN_NAME}
QA_RUN_FILE_PATH=${QA_RUN_PATH}/${QA_RUN_NAME}.run
QA_SPLIT_FILES=""
# python qa/create_split.py -i ${COLLECTION_PATH} -o ${DATASET_PATH}

for (( SPLIT=1; SPLIT<=${NUM_QA_SPLITS}; SPLIT++ )) do
    QA_SPLIT_RUN_NAME=HLTRI_COVID_LIES_QA_SPLIT_${SPLIT}_${RUN_ID}
    QA_SPLIT_RUN_MODEL_NAME=HLTRI_COVID_LIES_QA_SPLIT_${SPLIT}_${RUN_ID}
    QA_SPLIT_MODEL_NAME=qa-${DATASET}-${QA_SPLIT_RUN_MODEL_NAME}
    QA_SPLIT_PATH=${ARTIFACTS_PATH}/${QA_SPLIT_RUN_NAME}
    QA_SPLIT_FILE_PATH=${QA_SPLIT_PATH}/${QA_SPLIT_RUN_NAME}.qa

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
          --gpus ${GPUS}
    fi

    if [[ ${RUN_QA} = true ]]; then
        echo "Running split ${SPLIT} qa model..."
        python qa/qa_run.py \
          --split_path ${DATASET_PATH}/split_${SPLIT}.json \
          --pre_model_name ${QA_PRE_MODEL_NAME} \
          --model_name ${QA_SPLIT_MODEL_NAME} \
          --output_path ${QA_SPLIT_PATH} \
          --max_seq_len 128 \
          --batch_size 8 \
          --learning_rate 5e-5 \
          --epochs 20 \
          --load_trained_model \
          --gpus ${GPUS} \
        ; \
        python qa/format_qa.py \
          --input_path ${QA_SPLIT_PATH} \
          --output_path ${QA_SPLIT_FILE_PATH}
    fi
    QA_SPLIT_FILES="${QA_SPLIT_FILES},${QA_SPLIT_FILE_PATH}"
done

python qa/format_eval \
  --input_path ${QA_SPLIT_FILES} \
  --output_path ${QA_RUN_FILE_PATH} \
  --threshold ${QA_THRESHOLD}

#if [[ ${EVAL_QA} = true ]]; then
#    echo "Evaluating qa model..."
#    python qa/eval.py \
#      ${COLLECTION_PATH} \
#      ${QA_RUN_PATH} \
#      --task ${DATASET} \
#      > ${QA_EVAL_PATH} \
#      ; \
#      tail -n 3 ${QA_EVAL_PATH} \
#      | awk \
#        '{ for (i=1; i<=NF; i++) RtoC[i]= (RtoC[i]? RtoC[i] FS $i: $i) }
#        END{ for (i in RtoC) print RtoC[i] }' \
#      | tail -n 2
#fi
