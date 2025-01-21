# File: process_single_repo.py

import modal

import asyncio
import argparse
import json
import subprocess
import sys

from swebench.inference.make_datasets.create_instance import PATCH_EXAMPLE

from cognee.api.v1.cognify.code_graph_pipeline import run_code_graph_pipeline
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.modules.retrieval.description_to_codepart_search import (
    code_description_to_code_part_search,
)
from evals.eval_utils import download_github_repo


def check_install_package(package_name):
    """Check if a pip package is installed and install it if not."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return True
        except subprocess.CalledProcessError:
            return False


async def generate_patch_with_cognee(instance):
    repo_path = download_github_repo(instance, "../RAW_GIT_REPOS")
    include_docs = True
    problem_statement = instance["problem_statement"]
    instructions = read_query_prompt("patch_gen_kg_instructions.txt")

    async for result in run_code_graph_pipeline(repo_path, include_docs=include_docs):
        print(result)

    retrieved_codeparts = await code_description_to_code_part_search(
        problem_statement, include_docs=include_docs
    )

    prompt = "\n".join(
        [
            problem_statement,
            "<patch>",
            PATCH_EXAMPLE,
            "</patch>",
            "Additional context to solve the problem:",
            retrieved_codeparts,
        ]
    )

    llm_client = get_llm_client()
    answer_prediction = await llm_client.acreate_structured_output(
        text_input=prompt,
        system_prompt=instructions,
        response_model=str,
    )

    return answer_prediction


async def generate_patch_without_cognee(instance, llm_client):
    instructions = read_query_prompt("patch_gen_instructions.txt")

    print(instructions)
    print("-----")
    print(instance)
    answer_prediction = await llm_client.acreate_structured_output(
        text_input="dummy context",
        system_prompt=instructions,
        response_model=str,
    )
    return answer_prediction


async def process_repo(instance, volume, disable_cognee=True):
    """
    Process a single repository (a single instance).
    """
    print(instance)

    model_patch = await generate_patch_with_cognee(instance)
    model_name = "with_cognee"

    instance_name = instance["instance_id"]
    filename = f"{instance_name}_{model_name}patch.json"

    result = {
        "instance_id": instance["instance_id"],
        "model_patch": model_patch,
        "model_name_or_path": model_name,
    }

    with open(filename, "w") as f:
        json.dump(result, f, indent=4)
    with volume.batch_upload() as batch:
        batch.put_file(filename, filename)

    print(f"The repo {instance_name} was finished.")

    return None


async def main():
    """
    Main entry: expects a single repository (instance) in JSON form.
    Example usage:
      python process_single_repo.py --instance_json='{"instance_id": "abc123", ...}'
      or called as an imported function from Modal.
    """
    parser = argparse.ArgumentParser(description="Process a single repo from SWE-Bench")
    parser.add_argument("--instance_json", type=str, required=True)
    parser.add_argument(
        "--disable-cognee", action="store_true", help="Disable Cognee for evaluation"
    )
    args = parser.parse_args()

    # Install dependencies if needed
    for dependency in ["transformers", "sentencepiece", "swebench"]:
        check_install_package(dependency)

    # Parse the instance JSON from CLI
    instance = json.loads(args.instance_json)

    # Get the prediction
    print(instance)
    result = await process_repo(instance, disable_cognee=args.disable_cognee)

    # Construct a file name for the single result
    instance_id = instance["instance_id"]
    out_name = f"pred_{'nocognee' if args.disable_cognee else 'cognee'}_{instance_id}.json"

    with open(out_name, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Finished processing instance_id={instance_id}. Saved to {out_name}")


def entrypoint(instance, disable_cognee=True):
    asyncio.run(process_repo(instance, disable_cognee), debug=True)
