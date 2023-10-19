#!/bin/bash

# Set it to a specific GPU
export CUDA_VISIBLE_DEVICES=0

# Tasks and hyper-parameters
TASKS=( RTE MNLI MRPC QNLI  QQP )       # SST-2 and CoLA is not compatible with PastFuture
STEPS=(  50 2000   50  500 2000 )
FREEZ=(   1    3    2    1    2 )
BATCH=(  16   32   16   16   32 )
LEARN=(2e-5 3e-5 2e-5 1e-5 5e-5 )
GRADS=(   1    4    1    1    4 )
EPOCH=(   5    5    5    3    3 )
ENTRO=(0.25  0.2 0.06 0.10 0.09 )

# Base model
# - BERT   : bert / bert-base-uncased
BASETYPE=albert
BASENAME=albert-base-v2

# Project path
HOME_DIR=$(pwd)/..


# Run over the tasks
for tidx in "${!TASKS[@]}"; do

    # : run configurations
    TASKNAME=${TASKS[$tidx]}
    DATA_DIR=$HOME_DIR/datasets/originals/glue_hub/$TASKNAME
    DHUB_DIR=$HOME_DIR/datasets/originals/glue_hub_cache
    GLUEFILE=$HOME_DIR/project/glue.py
    RUN_OUTS=$HOME_DIR/model_output/$BASETYPE"_pastfuture_"$TASKNAME

    # : create folders to store the data
    mkdir -p $HOME_DIR/model_output
    mkdir -p $RUN_OUTS
    mkdir -p $HOME_DIR/datasets/originals/glue_hub_cache

    # : run eval
    echo "[finetune.pastfuture.sh] Run eval on the fine-tuned model"
    if [ "$BASETYPE" = "roberta" ]; then
        python -u run_glue_pastfuture.py \
            --model_type $BASETYPE \
            --model_name_or_path $RUN_OUTS \
            --task_name $TASKNAME \
            --data_dir "$DATA_DIR" \
            --max_seq_length 128 \
            --per_gpu_eval_batch_size=1 \
            --output_dir $RUN_OUTS \
            --hub_dir $DHUB_DIR \
            --logging_steps 1000 \
            --save_steps 1000 \
            --overwrite_cache \
            --do_eval \
            --patience ${ENTRO[$tidx]}

    else
        python -u run_glue_pastfuture.py \
            --model_type $BASETYPE \
            --model_name_or_path $RUN_OUTS \
            --task_name $TASKNAME \
            --data_dir "$DATA_DIR" \
            --max_seq_length 128 \
            --per_gpu_eval_batch_size=1 \
            --output_dir $RUN_OUTS \
            --hub_dir $DHUB_DIR \
            --logging_steps 1000 \
            --save_steps 1000 \
            --overwrite_cache \
            --do_lower_case \
            --do_eval \
            --patience ${ENTRO[$tidx]}

    fi

done
