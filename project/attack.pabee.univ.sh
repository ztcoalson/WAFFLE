#!/bin/bash

# set it to the first GPU
export CUDA_VISIBLE_DEVICES=0

# Tasks and hyper-parameters
# (universal attack only supports SST and SST-2 tasks)
TASKS=( SST-2 )
THRES=(     6 )

# Base model
# - BERT: bert / bert-base-uncased
BASETYPE=albert
BASENAME=albert-base-v2

# Project path
HOME_DIR=$(pwd)/..

# Attack configurations
ATTCK=( univ )              # Note: universal attacks
AGOAL=ours                  # Note: base, oavg, ours
AMETH=gradient              # Note: delete - fooler / gradient - a2t
ATHRE=( 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 )
LIMIT=1000
NWORD=3

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
            python attack_univ_exit.py \
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
                --attack_num_words=$NWORD \
                --multiexit \
                --num_examples 1000 \
        
        # :: Our attacks, use thresholds
        else
            for each_athres in ${ATHRE[@]}; do
                python attack_univ_exit.py \
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
                    --attack_num_words=$NWORD \
                    --multiexit \
                    --num_examples 1000 \

                exit    # for debugging

            done
        fi

    done

done
