"""Scripts to run QAWithDocSearch experiment
"""
import argparse
import os
import subprocess
from datetime import datetime
from importlib import import_module
from shutil import copyfile

import yaml

from config import ARTIFACT_DIR, DATA_DIR, EMBSTORE_DIR, LOGGER
from exp import QuestionAnsweringWithIndexSearchExperiment


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
    parser.add_argument(
        "--max_tokens_limit",
        "-mx",
        type=int,
        default=3375,
        help="Maximum Number of Tokens combined for all documents",
    )
    parser.add_argument(
        "--no_returned_docs",
        "-k",
        type=int,
        default=4,
        help="Number of documents to be returned from semantic search",
    )
    parser.add_argument(
        "--reduce_k_below_max_tokens",
        "-r",
        action="store_true",
        default=False,
        help="Whether to limit number of document tokens to below ",
    )
    parser.add_argument(
        "--chain_type",
        "-c",
        type=str,
        default="stuff",
        help="Type of chain to perform reduction operations",
    )
    parser.add_argument(
        "--iters", "-i", type=int, default=1, help="Number of iterations to run"
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

        for k, v in yaml_cfg.items():
            setattr(args, k, v)

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
    max_tokens_limit = args.max_tokens_limit
    no_returned_docs = args.no_returned_docs
    reduce_k_below_max_tokens = args.reduce_k_below_max_tokens
    chain_type = args.chain_type
    no_iters = args.iters

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
                os.path.join(DATA_DIR, "document_store", project),
                "--outputs",
                vectorstore_path,
                "--project",
                project,
                "--model",
                emb_type,
                "--chunk_size",
                str(chunk_size),
                "--chunk_overlap",
                str(chunk_overlap),
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

    experiment = QuestionAnsweringWithIndexSearchExperiment(
        prompt_template=prompt,
        vector_store=vectorstore_path,
        llm_type=llm_type,
        emb=emb_type,
        temperature=temperature,
        max_tokens=max_tokens,
        gt=os.path.join(DATA_DIR, "queries", ground_truth) if ground_truth else None,
        verbose=verbose,
        max_tokens_limit=max_tokens_limit,
        k=no_returned_docs,
        reduce_k_below_max_tokens=reduce_k_below_max_tokens,
    )

    LOGGER.info(
        "Successfully created experiment with settings:\n{}".format(
            "\n".join([f"{k}:{v}" for k, v in yaml_cfg.items()])
        )
    )
    print(f"Prompt:\n{experiment.convert_prompt_to_string(prompt)}")

    with open(test_case_path, "r") as f:
        test_cases = f.readlines()

    # experiment.load_json(os.path.join(ARTIFACT_DIR,
    #                                   "gpt-4_Chat_Tables_Chunk-size=1000_Overlap=200_Doc=10_Max-token=6500_Stuff_11-07-2023-21-48-04/result.json"))

    save_path = os.path.join(
        ARTIFACT_DIR,
        "{}_{}_{}".format(
            llm_type, description, datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        ),
    )
    os.makedirs(save_path, exist_ok=True)

    for idx in range(no_iters):
        experiment.reset()
        experiment.run_test_cases(
            test_cases,
            only_return_source=args.only_return_source,
            chain_type=chain_type,
        )
        LOGGER.info(f"Completed running all test cases for iteration {idx+1}.")

        experiment.save_json(os.path.join(save_path, f"result{idx+1}.json"))
        experiment.write_csv(
            os.path.join(save_path, f"result{idx+1}.csv"), num_docs=no_returned_docs
        )

    if args.yaml_cfg:
        copyfile(args.yaml_cfg, os.path.join(save_path, "settings.yaml"))
    else:
        settings = {
            "project": project,
            "test_case": test_case_path,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "description": description,
            "verbose": verbose,
            "emb_type": emb_type,
            "vectorstore": vectorstore,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "additional_docs": additional_docs,
            "pinecone_index_name": pinecone_index_name,
            "llm_type": llm_type,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_tokens_limit": max_tokens_limit,
            "k": k,
            "reduce_k_below_max_tokens": reduce_k_below_max_tokens,
            "chain_type": chain_type,
        }
        with open(os.path.join(save_path, "settings.yaml"), "w") as f:
            yaml.dump(settings, f)


if __name__ == "__main__":
    main()

# python3 ./src/scripts/qa_with_docs_search_experiment.py --yaml_cfg exps/uc_1.yaml
