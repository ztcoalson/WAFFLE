#!/bin/bash

# Set it to a specific GPU
export CUDA_VISIBLE_DEVICES=0

# Tasks and hyper-parameters
TASKS=( RTE MNLI MRPC QNLI  QQP CoLA )      # Note: MNLI is not working with bert-deebert
STEPS=(  50 2000   50  500 2000  200 )
FREEZ=(   1    3    2    1    2    2 )
BATCH=(  32   32   32   32   32   32 )
LEARN=(2e-5 3e-5 2e-5 1e-5 5e-5 1e-5 )
GRADS=(   1    4    1    1    4    1 )
EPOCH=(   5    5    5    5    5    5 )
ENTRO=(0.62 0.35  0.5 0.38 0.03 0.55 )      # Note: 1.5x speed-up

# Base model
BASETYPE=bert                               # bert
BASENAME=bert-base-uncased

# Project patch
HOME_DIR=$(pwd)/..


# Run over the tasks
for tidx in "${!TASKS[@]}"; do

    # : run configurations
    TASKNAME=${TASKS[$tidx]}
    DATA_DIR=$HOME_DIR/datasets/originals/glue_hub/$TASKNAME
    DHUB_DIR=$HOME_DIR/datasets/originals/glue_hub_cache
    GLUEFILE=$HOME_DIR/project/glue.py
    RUN_OUTS=$HOME_DIR/model_output/$BASETYPE"_deebert_"$TASKNAME/

    # : create folders to store the data
    mkdir -p $HOME_DIR/model_output
    mkdir -p $RUN_OUTS
    mkdir -p $HOME_DIR/datasets/originals/glue_hub_cache

    # : fine-tune
    echo "[finetune.deebert.sh] Fine-tune the model"
    if [ "$BASETYPE" = "roberta" ]; then
        python -u run_glue_deebert.py \
            --model_type $BASETYPE \
            --model_name_or_path $BASENAME \
            --task_name $TASKNAME \
            --do_train \
            --data_dir "$DATA_DIR" \
            --max_seq_length 128 \
            --per_gpu_eval_batch_size=1 \
            --per_gpu_train_batch_size=${BATCH[$tidx]} \
            --learning_rate ${LEARN[$tidx]} \
            --num_train_epochs ${EPOCH[$tidx]} \
            --overwrite_output_dir \
            --seed 42 \
            --output_dir $RUN_OUTS \
            --plot_data_dir $RUN_OUTS \
            --logging_steps 1000 \
            --save_steps 1000 \
            --hub_dir $DHUB_DIR \
            --overwrite_cache \
            --eval_after_first_stage

    else
        python -u run_glue_deebert.py \
            --model_type $BASETYPE \
            --model_name_or_path $BASENAME \
            --task_name $TASKNAME \
            --do_train \
            --do_lower_case \
            --data_dir "$DATA_DIR" \
            --max_seq_length 128 \
            --per_gpu_eval_batch_size=1 \
            --per_gpu_train_batch_size=${BATCH[$tidx]} \
            --learning_rate ${LEARN[$tidx]} \
            --num_train_epochs ${EPOCH[$tidx]} \
            --overwrite_output_dir \
            --seed 42 \
            --output_dir $RUN_OUTS \
            --plot_data_dir $RUN_OUTS \
            --logging_steps 1000 \
            --save_steps 1000 \
            --hub_dir $DHUB_DIR \
            --overwrite_cache \
            --eval_after_first_stage

    fi


    # : run eval
    echo "[finetune.deebert.sh] Run eval on the fine-tuned model"
    if [ "$BASETYPE" = "roberta" ]; then
        python -u run_glue_deebert.py \
            --model_type $BASETYPE \
            --model_name_or_path $RUN_OUTS \
            --task_name $TASKNAME \
            --data_dir "$DATA_DIR" \
            --max_seq_length 128 \
            --per_gpu_eval_batch_size=1 \
            --seed 42 \
            --output_dir $RUN_OUTS \
            --hub_dir $DHUB_DIR \
            --plot_data_dir $RUN_OUTS \
            --logging_steps 1000 \
            --save_steps 1000 \
            --overwrite_cache \
            --eval_after_first_stage \
            --do_eval \
            --eval_highway \
            --early_exit_entropy ${ENTRO[$tidx]} \

    else
        python -u run_glue_deebert.py \
            --model_type $BASETYPE \
            --model_name_or_path $RUN_OUTS \
            --task_name $TASKNAME \
            --do_lower_case \
            --data_dir "$DATA_DIR" \
            --max_seq_length 128 \
            --per_gpu_eval_batch_size=1 \
            --seed 42 \
            --output_dir $RUN_OUTS \
            --hub_dir $DHUB_DIR \
            --plot_data_dir $RUN_OUTS \
            --logging_steps 1000 \
            --save_steps 1000 \
            --overwrite_cache \
            --eval_after_first_stage \
            --do_eval \
            --eval_highway \
            --early_exit_entropy ${ENTRO[$tidx]} \

    fi

done
