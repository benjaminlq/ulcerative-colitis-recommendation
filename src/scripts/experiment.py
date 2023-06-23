import argparse
from exp import Experiment

from config import LOGGER

def get_argument_parser():
    parser = argparse.ArgumentParser("Run Experiment")
    parser.add_argument("--test_case", "-t", type=str, help="path to test case file")
    parser.add_argument("--prompt", "-p", type=str, help="path to prompt")
    parser.add_argument("--vectorstore", "-v", type=str, help="path to Vectorstore")
    args = parser.parse_args()
    return args
    
def main():
    args = get_argument_parser()
    
    experiment = Experiment(
        
    )

if __name__ == "__main__":
    main()
    
# python3 ./src/scripts/experiment.py