#!/bin/bash

for CFG_FILE in "$@"
    do
        EXP_PATH="exps/$CFG_FILE"
        echo "Running Experiment Script on cfg file $EXP_PATH"
        python3 ./src/scripts/qa_with_docs_search_experiment.py --yaml_cfg $EXP_PATH
    done

# ./src/bash/multi_denseemb_exps.sh uc_12_chat.yaml uc_13_chat.yaml uc_14_chat.yaml