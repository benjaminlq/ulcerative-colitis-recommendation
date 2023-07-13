"""Scripts to run experiment
"""
import argparse
import json
import os
from datetime import datetime
from importlib import import_module

import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import ARTIFACT_DIR, DATA_DIR, DOCUMENT_SOURCE, EXCLUDE_DICT, LOGGER
from exp import MapReduceQAOverDocsExperiment
from utils import convert_csv_to_documents, convert_json_to_documents, load_documents


def get_argument_parser():
    """Argument Parser

    Returns:
        args: argument dictionary
    """
    parser = argparse.ArgumentParser("Run Experiment")

    # Project Level Arguments
    parser.add_argument("--yaml_cfg", "-y", type=str, help="path_to_yaml_file")
    parser.add_argument("--project", "-j", type=str, default=None, help="project")
    parser.add_argument(
        "--test_case", "-t", type=str, default=None, help="path to test case file"
    )
    parser.add_argument("--prompt", "-p", type=str, default=None, help="path to prompt")
    parser.add_argument("--description", "-d", type=str, default="", help="description")
    parser.add_argument(
        "--verbose",
        "-vb",
        type=bool,
        default=True,
        help="Verbose settings for Langchain",
    )

    # Prompt & LLM Settings
    parser.add_argument("--temperature", type=float, default=0, help="LLM temperature")
    parser.add_argument(
        "--chain_type",
        "-c",
        type=str,
        default="map_reduce",
        help="Type of chain to perform reduction operations",
    )
    parser.add_argument(
        "--llm_type", type=str, default="gpt-3.5-turbo", help="Type of MAP LLM model"
    )
    parser.add_argument(
        "--max_gen_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to be generated by MAP LLM",
    )
    parser.add_argument(
        "--reduce_llm", type=str, default="gpt-3.5-turbo-16k", help="Type of LLM model"
    )
    parser.add_argument(
        "--combine_max_gen_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to be generated by REDUCE LLM",
    )
    parser.add_argument(
        "--collapse_llm",
        type=str,
        default="gpt-3.5-turbo-16k",
        help="Type of LLM model",
    )
    parser.add_argument(
        "--collapse_max_gen_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to be generated by COLLAPSE LLM",
    )
    parser.add_argument(
        "--combine_max_doc_tokens",
        type=int,
        default=14000,
        help="Maximum number of tokens to be generated by COLLAPSE LLM",
    )
    parser.add_argument(
        "--collapse_max_doc_tokens",
        type=int,
        default=6000,
        help="Maximum number of tokens to be generated by COLLAPSE LLM",
    )

    # Document Settings
    parser.add_argument("--document_level", type=str, default="page", help="page|chunk")
    parser.add_argument(
        "--chunk_size", "-cs", type=int, default=5000, help="Chunk Size"
    )
    parser.add_argument(
        "--chunk_overlap", "-co", type=int, default=500, help="Chunk Overlap"
    )
    parser.add_argument(
        "--ground_truth",
        "-gt",
        type=str,
        default=None,
        help="Path to ground truth file",
    )
    parser.add_argument(
        "--additional_docs", type=str, default=None, help="Path to additional documents"
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
    description = args.description
    verbose = args.verbose

    temperature = args.temperature
    llm_type = args.llm_type
    max_gen_tokens = args.max_gen_tokens
    reduce_llm = args.reduce_llm
    combine_max_gen_tokens = args.combine_max_gen_tokens
    collapse_llm = args.collapse_llm
    collapse_max_gen_tokens = args.collapse_max_gen_tokens
    combine_max_doc_tokens = args.combine_max_doc_tokens
    collapse_max_doc_tokens = args.collapse_max_doc_tokens

    document_level = args.document_level
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    ground_truth = args.ground_truth
    additional_docs = args.additional_docs

    map_prompt = import_module(
        f"prompts.{project}.{os.path.splitext(prompt)[0]}"
    ).CHAT_QUESTION_PROMPT

    combine_prompt = import_module(
        f"prompts.{project}.{os.path.splitext(prompt)[0]}"
    ).CHAT_COMBINE_PROMPT

    try:
        collapse_prompt = import_module(
            f"prompts.{project}.{os.path.splitext(prompt)[0]}"
        ).CHAT_COLLAPSE_PROMPT
    except ImportError:
        collapse_prompt = None

    experiment = MapReduceQAOverDocsExperiment(
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        collapse_prompt=collapse_prompt,
        llm_type=llm_type,
        reduce_llm=reduce_llm,
        collapse_llm=collapse_llm,
        temperature=temperature,
        max_gen_tokens=max_gen_tokens,
        combine_max_gen_tokens=combine_max_gen_tokens,
        collapse_max_gen_tokens=collapse_max_gen_tokens,
        combine_max_doc_tokens=combine_max_doc_tokens,
        collapse_max_doc_tokens=collapse_max_doc_tokens,
        gt=os.path.join(DATA_DIR, "queries", ground_truth) if ground_truth else None,
        verbose=verbose,
    )

    LOGGER.info(
        "Successfully created experiment with settings:\n{}".format(
            "\n".join([f"{k}:{v}" for k, v in vars(args).items()])
        )
    )

    with open(test_case_path, "r") as f:
        test_cases = f.readlines()

    documents = load_documents(
        os.path.join(DOCUMENT_SOURCE, project), exclude_pages=EXCLUDE_DICT
    )

    if document_level == "chunk":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        documents = text_splitter.split_documents(documents)

    if additional_docs:
        with open(additional_docs, "r") as f:
            add_doc_infos = json.load(f)
        for add_doc_info in add_doc_infos:
            if add_doc_info["mode"] == "table":
                documents.extend(
                    convert_csv_to_documents(add_doc_info, concatenate_rows=True)
                )
            elif add_doc_info["mode"] == "json":
                documents.extend(convert_json_to_documents(add_doc_info))
            else:
                LOGGER.warning(
                    "Invalid document type. No texts added to documents list"
                )

    # experiment.load_json(
    #     os.path.join(
    #         ARTIFACT_DIR,
    #         "Map_Reduce_Chunk_Stuff_5000_500_gpt-3.5_13-07-2023-17-01-28/result.json"
    #     )
    # )

    experiment.run_test_cases(
        test_cases, docs=documents[:2], return_intermediate_steps=True
    )

    LOGGER.info("Completed running all test cases.")

    save_path = os.path.join(
        ARTIFACT_DIR,
        "{}_{}".format(description, datetime.now().strftime("%d-%m-%Y-%H-%M-%S")),
    )
    # save_path = os.path.join(
    #         ARTIFACT_DIR,
    #         "Map_Reduce_Chunk_Stuff_5000_500_gpt-3.5_13-07-2023-17-01-28"
    #     )

    os.makedirs(save_path, exist_ok=True)
    experiment.save_json(os.path.join(save_path, "result.json"))
    experiment.write_csv(os.path.join(save_path, "result.csv"))

    settings = vars(args)
    with open(os.path.join(save_path, "settings.yaml"), "w") as f:
        yaml.dump(settings, f)


if __name__ == "__main__":
    main()

# python3 ./src/scripts/qa_over_docs_experiment.py --yaml_cfg exps/map_reduce_1.yaml
