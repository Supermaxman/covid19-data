#!/usr/bin/env bash

export QA_SPLIT=1
# run names
export QA_RUN_NAME=HLTRI_COVID_LIES_QA_SPLIT_${QA_SPLIT}_2
export QA_RUN_MODEL_NAME=HLTRI_COVID_LIES_QA_SPLIT_${QA_SPLIT}_2

# collection
export DATASET=covid-lies

# major hyper-parameters for system
export QA_PRE_MODEL_NAME=digitalepidemiologylab/covid-twitter-bert-v2
#export QA_PRE_MODEL_NAME=nboost/pt-biobert-base-msmarco


# qa flags
# QA fine-tune qaing model using training set
export TRAIN_QA=true
# QA run qa using trained model on validation set
export RUN_QA=true
# QA run evaluation script on validation set
export EVAL_QA=true

export QA_MODEL_NAME=qa-${DATASET}-${QA_RUN_MODEL_NAME}
export DATASET_PATH=data
export COLLECTION_PATH=${DATASET_PATH}/downloaded_tweets_labeled.jsonl
export SPLIT_PATH=${DATASET_PATH}/split_${QA_SPLIT}.json

export ARTIFACTS_PATH=artifacts/${DATASET}

export QA_PATH=${ARTIFACTS_PATH}/${QA_RUN_NAME}
export QA_FILE_PATH=${QA_PATH}/${QA_RUN_NAME}.qa

export QA_RUN_PATH=${QA_PATH}/${QA_RUN_NAME}.txt
export QA_EVAL_PATH=${QA_PATH}/${QA_RUN_NAME}.eval




if [[ ${TRAIN_QA} = true ]]; then
    # python qa/create_split.py -i ${COLLECTION_PATH} -o ${DATASET_PATH}
    echo "Training qa model..."
    python qa/qa_train.py \
      --split_path ${SPLIT_PATH} \
      --pre_model_name ${QA_PRE_MODEL_NAME} \
      --model_name ${QA_MODEL_NAME} \
      --max_seq_len 128 \
      --batch_size 8 \
      --learning_rate 5e-6 \
      --epochs 20 \
      --gpus 3,4,5,6
fi

#if [[ ${RUN_QA} = true ]]; then
#    echo "Running qa model..."
#    python qa/qa_run.py \
#      --collection_path ${COLLECTION_PATH} \
#      --label_path ${LABEL_PATH} \
#      --output_path ${QA_PATH} \
#      --pre_model_name ${QA_PRE_MODEL_NAME} \
#      --model_name ${QA_MODEL_NAME} \
#      --max_seq_len 96 \
#      --load_trained_model \
#    ; \
#    python qa/format_qa.py \
#      --input_path ${QA_PATH} \
#      --output_path ${QA_FILE_PATH} \
#    ; \
#    python qa/format_eval \
#      --input_path ${QA_FILE_PATH} \
#      --output_path ${QA_RUN_PATH} \
#      --top_k 1000
#
#    python -m qa.extract_answers \
#      --search_path ${QA_RUN_PATH} \
#      --collection_path ${COLLECTION_PATH} \
#      --output_path ${ANSWERS_PATH}
#fi


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
