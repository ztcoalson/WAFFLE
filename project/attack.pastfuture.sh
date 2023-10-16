#!/bin/bash

# set it to a specific GPU
export CUDA_VISIBLE_DEVICES=0

#may be case sensitive for directories
TASKS=( RTE MNLI MRPC QNLI  QQP )       # SST-2 and CoLA is not compatible with PastFuture code
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
    RUN_OUTS=$HOME_DIR/model_output/$BASETYPE"_pastfuture_"$TASKNAME
    RES_OUTS=$HOME_DIR/results/$BASETYPE"_pastfuture_"$TASKNAME

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
                --entropy ${ENTRO[$tidx]} \
                --regression_threshold 0.1 \
                --exit_type "pastfuture" \
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
                    --entropy ${ENTRO[$tidx]} \
                    --regression_threshold 0.1 \
                    --exit_type "pastfuture" \
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
