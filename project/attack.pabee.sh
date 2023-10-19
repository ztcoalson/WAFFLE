#!/bin/bash

# set it to the first GPU
export CUDA_VISIBLE_DEVICES=1

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

# Attack configurations
ATTCK=( a2t )               # Note: fooler or a2t
AGOAL=base                  # Note: base, oavg, ours
AMETH=gradient              # Note: delete - fooler / gradient - a2t
ATHRE=( 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 )
LIMIT=1000

# Run over the tasks and attacks
for tidx in "${!TASKS[@]}"; do

    # : run configurations
    TASKNAME=${TASKS[$tidx]}
    DATA_DIR=$HOME_DIR/datasets/originals/glue_hub/$TASKNAME
    DHUB_DIR=$HOME_DIR/datasets/originals/glue_hub_cache
    GLUEFILE=$HOME_DIR/project/glue.py
    RUN_OUTS=$HOME_DIR/model_output/$BASETYPE"_PABEE_"$TASKNAME
    RES_OUTS=$HOME_DIR/results/$BASETYPE"_PABEE_"$TASKNAME

    # : create folders to store the data
    mkdir -p $HOME_DIR/model_output
    mkdir -p $RUN_OUTS
    mkdir -p $RES_OUTS
    mkdir -p $HOME_DIR/datasets/originals/glue_hub_cache

    # : Run over the attacks
    for each_attack in "${ATTCK[@]}"; do

        # :: Baselines, no threshold
        if [[ ("$AGOAL" == "base") || ("$AGOAL" == "oavg") ]]; then
            python attack_exit.py \
                --model_type $BASETYPE \
                --model_path $RUN_OUTS \
                --model_name $BASENAME \
                --task_name $TASKNAME \
                --data_dir "$DATA_DIR" \
                --result_dir $RES_OUTS \
                --max_seq_length 128 \
                --eval_reg_threshold 0.5 \
                --patience ${THRES[$tidx]} \
                --regression_threshold 0.1 \
                --exit_type "pabee" \
                --glue_script $GLUEFILE \
                --attack "$each_attack" \
                --attack_goal=$AGOAL \
                --attack_method=$AMETH \
                --attack_maxquery=$LIMIT \
                --multiexit \
                --num_examples 1000 \
        
        # :: Our attacks, use thresholds
        else
            for each_athres in ${ATHRE[@]}; do
                python attack_exit.py \
                    --model_type $BASETYPE \
                    --model_path $RUN_OUTS \
                    --model_name $BASENAME \
                    --task_name $TASKNAME \
                    --data_dir "$DATA_DIR" \
                    --result_dir $RES_OUTS \
                    --max_seq_length 128 \
                    --eval_reg_threshold 0.5 \
                    --patience ${THRES[$tidx]} \
                    --regression_threshold 0.1 \
                    --exit_type "pabee" \
                    --glue_script $GLUEFILE \
                    --attack "$each_attack" \
                    --attack_goal=$AGOAL \
                    --attack_method=$AMETH \
                    --attack_maxquery=$LIMIT \
                    --attack_threshold=$each_athres \
                    --multiexit \
                    --num_examples 1000 \

            done
        fi

    done

done
