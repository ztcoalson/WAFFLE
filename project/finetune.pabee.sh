#!/bin/bash

# Set it to the first GPU
export CUDA_VISIBLE_DEVICES=0

# Tasks and hyper-parameters
TASKS=( RTE MNLI MRPC QNLI  QQP SST-2 CoLA )
STEPS=(  50 2000   50  500 2000   200  200 )
FREEZ=(   1    3    2    1    2     2    2 )
BATCH=(  32  128  128  128   32   128  128 )
LEARN=(3e-5 1e-5 4e-5 1e-5 2e-5  2e-5 4e-5 )
GRADS=(   1    4    1    1    4     1    1 )
EPOCH=(   5    5    5    5    5     3    5 )
THRES=(   6    6    6    6    6     6    6 )

# Base model
# - BERT: bert / bert-base-uncased
BASETYPE=albert
BASENAME=albert-base-v2

# Project path
HOME_DIR=$(pwd)/..

# Run over the tasks
for tidx in "${TASKS[@]}"; do

    # : run configurations
    TASKNAME=${TASKS[$tidx]}
    DATA_DIR=$HOME_DIR/datasets/originals/glue_hub/$TASKNAME
    DHUB_DIR=$HOME_DIR/datasets/originals/glue_hub_cache
    GLUEFILE=$HOME_DIR/project/glue.py
    RUN_OUTS=$HOME_DIR/model_output/$BASETYPE"_pabee_"$TASKNAME"_"${LEARN[$tidx]}"_"${BATCH[$tidx]}/

    # : create folders to store the data
    mkdir -p $HOME_DIR/model_output
    mkdir -p $RUN_OUTS
    mkdir -p $HOME_DIR/datasets/originals/glue_hub_cache

    # : fine-tune
    echo "[finetune.pabee.sh] Fine-tune the model"
    python ./run_glue_pabee.py \
        --model_type $BASETYPE \
        --model_name_or_path $BASENAME \
        --task_name $TASKNAME \
        --do_train \
        --do_lower_case \
        --data_dir "$DATA_DIR" \
        --max_seq_length 128 \
        --per_gpu_train_batch_size=${BATCH[$tidx]} \
        --per_gpu_eval_batch_size=${BATCH[$tidx]} \
        --learning_rate ${LEARN[$tidx]} \
        --save_steps 1000 \
        --logging_steps 1000 \
        --num_train_epochs ${EPOCH[$tidx]} \
        --output_dir $RUN_OUTS \
        --hub_dir $DHUB_DIR \
        --overwrite_output_dir \
        --overwrite_cache \
        --glue_script $GLUEFILE \

    # : run eval
    echo "[finetune.pabee.sh] Run eval on the fine-tuned model"
    python run_glue_pabee.py \
        --model_type $BASETYPE \
        --model_name_or_path $RUN_OUTS \
        --task_name $TASKNAME \
        --do_eval \
        --do_lower_case \
        --data_dir "$DATA_DIR" \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size 1 \
        --logging_steps 1000 \
        --output_dir $RUN_OUTS \
        --hub_dir $DHUB_DIR \
        --patience ${THRES[$tidx]} \
        --glue_script $GLUEFILE \

done
