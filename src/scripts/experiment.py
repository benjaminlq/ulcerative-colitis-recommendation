import argparse

from config import LOGGER
from exp import Experiment
import os
from importlib import import_module

def get_argument_parser():
    parser = argparse.ArgumentParser("Run Experiment")
    parser.add_argument("--project", "-j", type=str, help="project")
    parser.add_argument("--test_case", "-t", type=str, help="path to test case file")
    parser.add_argument("--prompt", "-p", type=str, help="path to prompt")
    parser.add_argument("--vectorstore", "-v", type=str, help="path to Vectorstore")
    args = parser.parse_args()
    return args


def main():
    args = get_argument_parser()
    project = args.project
    test_case_path = args.test_case
    prompt = args.prompt
    vectorstore_path = args.vectorstore
    if args.yaml_cfg:
        # Load yaml file
        project = 
        test_case_path = 
        prompt_path =
        vectorstore_path = 

    with open(test_case_path, "r") as f:
        test_cases = f.readlines()
    
    prompt = import_module()
    

    experiment = Experiment()


if __name__ == "__main__":
    main()

# python3 ./src/scripts/experiment.py
