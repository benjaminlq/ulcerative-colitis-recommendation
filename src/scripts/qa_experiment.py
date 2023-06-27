"""Scripts to run experiment
"""
import argparse
import os
import subprocess
from datetime import datetime
from importlib import import_module

import yaml

from config import ARTIFACT_DIR, DATA_DIR, EMBSTORE_DIR, LOGGER
from exp import Experiment


def get_argument_parser():
    """Argument Parser

    Returns:
        args: argument dictionary
    """
    parser = argparse.ArgumentParser("Run Experiment")
    parser.add_argument("--yaml_cfg", "-y", type=str, help="path_to_yaml_file")
    parser.add_argument("--project", "-j", type=str, default=None, help="project")
    parser.add_argument(
        "--test_case", "-t", type=str, default=None, help="path to test case file"
    )
    parser.add_argument("--prompt", "-p", type=str, default=None, help="path to prompt")
    parser.add_argument(
        "--description",
        "-d",
        type=str,
        default="",
        help="description of the experiment",
    )

    parser.add_argument(
        "--llm_type", "-m", type=str, default="gpt-3.5-turbo", help="Type of LLM model"
    )
    parser.add_argument(
        "--temperature", "-tp", type=float, default=0, help="LLM temperature"
    )
    parser.add_argument(
        "--max_tokens",
        "-tk",
        type=int,
        default=512,
        help="Maximum number of tokens to be generated",
    )

    parser.add_argument(
        "--vectorstore", "-v", type=str, default=None, help="path to vectorstore"
    )
    parser.add_argument(
        "--emb_type",
        "-e",
        type=str,
        default="text-embedding-ada-002",
        help="Type of Embeddings model",
    )
    parser.add_argument(
        "--chunk_size", "-cs", type=int, default=1000, help="Chunk Size"
    )
    parser.add_argument(
        "--chunk_overlap", "-co", type=int, default=50, help="Chunk Overlap"
    )
    parser.add_argument(
        "--pinecone_index_name",
        "-pc",
        type=str,
        default=None,
        help="pinecone_index_name",
    )

    parser.add_argument(
        "--ground_truth",
        "-gt",
        type=str,
        default=None,
        help="Path to ground truth file",
    )
    parser.add_argument(
        "--verbose",
        "-vb",
        type=bool,
        default=True,
        help="Verbose settings for Langchain",
    )
    parser.add_argument(
        "--additional_docs",
        "-a",
        type=str,
        default=None,
        help="Path to additional documents",
    )
    parser.add_argument(
        "--only_return_source",
        action="store_true",
        default=False,
        help="Only return source documents from semantic search",
    )
    args = parser.parse_args()
    return args


def main():
    """Main Execution Function"""
    args = get_argument_parser()

    if args.yaml_cfg:
        # Load yaml file
        with open(args.yaml_cfg, "r") as f:
            yaml_cfg = yaml.safe_load(f)
        project = yaml_cfg["project"]
        test_case_path = yaml_cfg["test_case"]
        prompt = yaml_cfg["prompt"]
        vectorstore = yaml_cfg["vectorstore"]
        llm_type = yaml_cfg["llm_type"]
        emb_type = yaml_cfg["emb_type"]
        temperature = yaml_cfg["temperature"]
        max_tokens = yaml_cfg["max_tokens"]
        ground_truth = yaml_cfg["ground_truth"]
        verbose = yaml_cfg["verbose"]
        chunk_size = yaml_cfg["chunk_size"]
        chunk_overlap = yaml_cfg["chunk_overlap"]
        pinecone_index_name = yaml_cfg["pinecone_index_name"]
        description = yaml_cfg["description"]
        additional_docs = yaml_cfg["additional_docs"]

    else:
        project = args.project
        test_case_path = args.test_case
        prompt = args.prompt
        vectorstore = args.vectorstore
        llm_type = args.llm_type
        emb_type = args.emb_type
        temperature = args.temperature
        max_tokens = args.max_tokens
        ground_truth = args.ground_truth
        verbose = args.verbose
        chunk_size = args.chunk_size
        chunk_overlap = args.chunk_overlap
        pinecone_index_name = args.pinecone_index_name
        description = args.description
        additional_docs = args.additional_docs

    assert project is not None, "Project not specified"
    assert test_case_path is not None, "Test Case Path is not specified"
    assert prompt is not None, "Prompt is not specified"
    assert vectorstore is not None, "Vectorstore is not specified"

    vectorstore_path = os.path.join(EMBSTORE_DIR, project, vectorstore)
    if not os.path.exists(vectorstore_path):
        p = subprocess.run(
            [
                "python3",
                "ingest.py",
                "--embed_store",
                vectorstore.split("/")[0],
                "--inputs",
                os.path.join(DATA_DIR, project),
                "--outputs",
                vectorstore_path,
                "--project",
                project,
                "--model",
                emb_type,
                "--chunk_size",
                chunk_size,
                "--chunk_overlap",
                chunk_overlap,
                "--pinecone_index_name",
                pinecone_index_name,
                "--additional_docs",
                additional_docs,
            ]
        )
        LOGGER.info(
            f"Process successfully executed with code {p.returncode}"
            if p.returncode == 0
            else f"Process unsuccessfully executed with code {p.returncode}"
        )

    prompt = import_module(
        f"prompts.{project}.{os.path.splitext(prompt)[0]}"
    ).PROMPT_TEMPLATE

    experiment = Experiment(
        prompt_template=prompt,
        vector_store=vectorstore_path,
        llm_type=llm_type,
        emb=emb_type,
        temperature=temperature,
        max_tokens=max_tokens,
        gt=os.path.join(DATA_DIR, "queries", ground_truth) if ground_truth else None,
        verbose=verbose,
    )

    LOGGER.info(
        "Successfully created experiment with settings:\n{}".format(
            "\n".join([f"{k}:{v}" for k, v in yaml_cfg.items()])
        )
    )

    with open(test_case_path, "r") as f:
        test_cases = f.readlines()

    experiment.run_test_cases(test_cases, only_return_source=args.only_return_source)
    LOGGER.info("Completed running all test cases.")

    save_path = os.path.join(
        ARTIFACT_DIR,
        "{}_{}_{}".format(
            llm_type, description, datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
        ),
    )
    experiment.save_json(save_path + ".json")
    experiment.write_csv(save_path + ".csv")


if __name__ == "__main__":
    main()

# python3 ./src/scripts/qa_experiment.py --yaml_cfg exps/uc_1.yaml
