#!/bin/bash

# Set it to the first GPU
export CUDA_VISIBLE_DEVICES=0

# Tasks and hyper-parameters
TASKS=( RTE SST-2 )
STEPS=(  50   200 )
FREEZ=(   1     2 )
BATCH=(  32   128 )
LEARN=(3e-5  2e-5 )
GRADS=(   1     1 )
EPOCH=(   5     3 )
THRES=(   6     6 )

# Base model
BASETYPE=albert
BASENAME=albert-base-v2

# Project path
HOME_DIR=$(pwd)/..

# AT
ATT_GOAL=( base )                               # Note: base or ours

# Run over the goals and tasks
for goal in "${ATT_GOAL[@]}"; do
    for tidx in "${!TASKS[@]}"; do

        # : run configurations
        TASKNAME=${TASKS[$tidx]}
        DATA_DIR=$HOME_DIR/datasets/originals/glue_hub/$TASKNAME
        DHUB_DIR=$HOME_DIR/datasets/originals/glue_hub_cache
        GLUEFILE=$HOME_DIR/project/glue.py
        RUN_OUTS=$HOME_DIR/model_output/$BASETYPE"_PABEE_"$TASKNAME"_"${LEARN[$tidx]}"_"${BATCH[$tidx]}"_at_"$goal/

        # : create folders to store the data
        mkdir -p $HOME_DIR/model_output
        mkdir -p $RUN_OUTS
        mkdir -p $HOME_DIR/datasets/originals/glue_hub_cache

        # : fine-tune
        echo "[finetune.pabee.sh] Fine-tune the model (adversarial training)"
        python ./at_run_glue_pabee.py \
            --model_type $BASETYPE \
            --model_name_or_path $BASENAME \
            --task_name $TASKNAME \
            --do_train \
            --do_lower_case \
            --data_dir "$DATA_DIR" \
            --max_seq_length 128 \
            --train_batch_size=${BATCH[$tidx]} \
            --valid_batch_size=${BATCH[$tidx]} \
            --learning_rate ${LEARN[$tidx]} \
            --save_steps 1000 \
            --logging_steps 1000 \
            --num_train_epochs ${EPOCH[$tidx]} \
            --output_dir $RUN_OUTS \
            --hub_dir $DHUB_DIR \
            --overwrite_output_dir \
            --overwrite_cache \
            --glue_script $GLUEFILE \
            --adv_train \
            --attack_goal $goal \
            --multiexit

        # : run eval
        echo "[finetune.pabee.sh] Run eval on the fine-tuned, robust model"
        python at_run_glue_pabee.py \
            --model_type $BASETYPE \
            --model_name_or_path $RUN_OUTS \
            --task_name $TASKNAME \
            --do_eval \
            --do_lower_case \
            --data_dir "$DATA_DIR" \
            --max_seq_length 128 \
            --valid_batch_size 1 \
            --logging_steps 1000 \
            --output_dir $RUN_OUTS \
            --hub_dir $DHUB_DIR \
            --patience ${THRES[$tidx]} \
            --glue_script $GLUEFILE \
            --adv_train \
            --attack_goal $goal \
            --multiexit

    done
done
