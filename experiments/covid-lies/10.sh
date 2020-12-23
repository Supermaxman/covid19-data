#!/usr/bin/env bash

# run names
RUN_ID=10

# collection
DATASET=covid-lies

# major hyper-parameters for system
QA_PRE_MODEL_NAME=digitalepidemiologylab/covid-twitter-bert-v2
#export QA_PRE_MODEL_NAME=nboost/pt-biobert-base-msmarco
QA_THRESHOLD=0.1

QA_BATCH_SIZE=8
QA_MAX_SEQ_LEN=128
QA_TRAIN_GPUS=4
QA_EVAL_GPUS=4


NUM_QA_SPLITS=5
# qa flags
# QA fine-tune qaing model using training set
TRAIN_QA=true
# QA run qa using trained model on validation set
RUN_QA=true
# QA run evaluation script on validation set
EVAL_QA=true

DATASET_PATH=data
ARTIFACTS_PATH=artifacts/${DATASET}
QA_SPLIT_FILES=""

# python qa/create_split.py -i ${COLLECTION_PATH} -o ${DATASET_PATH}

for (( SPLIT=1; SPLIT<=${NUM_QA_SPLITS}; SPLIT++ )) do
    if [[ ${TRAIN_QA} = true ]]; then
        echo "Training split ${SPLIT} qa model..."
        python qa/qa_train.py \
          --split_path ${DATASET_PATH}/split_${SPLIT}.json \
          --pre_model_name ${QA_PRE_MODEL_NAME} \
          --model_name qa-${DATASET}-HLTRI_COVID_LIES_QA_SPLIT_${SPLIT}_${RUN_ID} \
          --hera_path ${DATASET_PATH}/all_tweets_labeled_hera.json \
          --max_seq_len ${QA_MAX_SEQ_LEN} \
          --batch_size ${QA_BATCH_SIZE} \
          --learning_rate 5e-5 \
          --epochs 20 \
          --fine_tune \
          --gpus ${QA_TRAIN_GPUS}
    fi

    if [[ ${RUN_QA} = true ]]; then
        echo "Running split ${SPLIT} qa..."
        python qa/qa_predict.py \
          --split_path ${DATASET_PATH}/split_${SPLIT}.json \
          --pre_model_name ${QA_PRE_MODEL_NAME} \
          --model_name qa-${DATASET}-HLTRI_COVID_LIES_QA_SPLIT_${SPLIT}_${RUN_ID} \
          --output_path ${ARTIFACTS_PATH}/HLTRI_COVID_LIES_QA_SPLIT_${SPLIT}_${RUN_ID} \
          --max_seq_len ${QA_MAX_SEQ_LEN} \
          --batch_size ${QA_BATCH_SIZE} \
          --load_trained_model \
          --gpus ${QA_EVAL_GPUS} \
        ; \
        python qa/format_qa_predictions.py \
          --input_path ${ARTIFACTS_PATH}/HLTRI_COVID_LIES_QA_SPLIT_${SPLIT}_${RUN_ID} \
          --output_path ${ARTIFACTS_PATH}/HLTRI_COVID_LIES_QA_SPLIT_${SPLIT}_${RUN_ID}/predictions.qa
    fi

    QA_SPLIT_FILES="${QA_SPLIT_FILES},${ARTIFACTS_PATH}/HLTRI_COVID_LIES_QA_SPLIT_${SPLIT}_${RUN_ID}/predictions.qa"
done

if [[ ${EVAL_QA} = true ]]; then
    echo "Evaluating qa model..."
    mkdir -p ${ARTIFACTS_PATH}/HLTRI_COVID_LIES_QA_${RUN_ID}
    python qa/format_qa_eval.py \
      --input_path ${QA_SPLIT_FILES} \
      --output_path ${ARTIFACTS_PATH}/HLTRI_COVID_LIES_QA_${RUN_ID}/all.run \
      --threshold ${QA_THRESHOLD}

    python qa/qa_eval.py \
      --label_path ${DATASET_PATH}/downloaded_tweets_labeled.jsonl \
      --run_path ${ARTIFACTS_PATH}/HLTRI_COVID_LIES_QA_${RUN_ID}/all.run \
      > ${ARTIFACTS_PATH}/HLTRI_COVID_LIES_QA_${RUN_ID}/eval.txt \
      ; \
      cat ${ARTIFACTS_PATH}/HLTRI_COVID_LIES_QA_${RUN_ID}/eval.txt
fi


# TODO expand with HERA
