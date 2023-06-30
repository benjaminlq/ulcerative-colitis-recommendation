#!/bin/bash

for CFG_FILE in "$@"
    do
        EXP_PATH="exps/$CFG_FILE"
        echo "Running Experiment Script on cfg file $EXP_PATH"
        python3 ./src/scripts/qa_experiment.py --yaml_cfg $EXP_PATH
    done

# ./src/bash/multi_exps.sh uc_4.yaml uc_5.yaml uc_6.yaml uc_7.yaml