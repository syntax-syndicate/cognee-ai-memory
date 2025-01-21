import subprocess
import json
import os
import asyncio
from swebench.harness.utils import load_swebench_dataset
import dotenv

dotenv.load_dotenv()


async def run_in_docker(instance, instance_id, disable_cognee, dockerfile_path):
    instance_json = json.dumps(instance)

    # Check if Docker is installed and running
    try:
        subprocess.run(
            ["docker", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print("Docker is installed.")
    except subprocess.CalledProcessError:
        print("Docker is not installed or not running. Please install and start Docker.")
        return None

    # Define the container image name
    docker_image_name = "swebench-processor-local"

    # Build the Docker image from the local Dockerfile (if not already built)
    print(f"Building the Docker image {docker_image_name} from local Dockerfile...")
    build_process = subprocess.run(
        ["docker", "build", "-t", docker_image_name, "-f", dockerfile_path, "."],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if build_process.returncode != 0:
        print("Failed to build Docker image. Check the Dockerfile for errors.")
        return None
    print("Docker image built successfully.")

    # Run the Docker container locally with the instance data and capture output
    command = [
        "docker",
        "run",
        "--rm",
        "-e",
        f"INSTANCE_DATA={instance_json}",
        "-e",
        f"DISABLE_COGNEE={disable_cognee}",
        "-e",
        f"LLM_API_KEY={os.getenv('LLM_API_KEY')}",
        docker_image_name,
    ]

    print(f"Launching container for instance {instance_id}...")

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,  # Capture stdout
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await process.communicate()

    if process.returncode == 0:
        output = stdout.decode().strip()
        print(f"Container {instance_id} output: {output}")
        return output  # Returning the processed instance data from stdout
    else:
        print(f"Container {instance_id} failed. Error: {stderr.decode()}")
        return None


async def main(disable_cognee=True, num_samples=2, dockerfile_path="Dockerfile_SWE"):
    print("Configuration:")
    print("- Running locally with Docker")
    print(f"- Disable Cognee: {disable_cognee}")
    print(f"- Number of samples: {num_samples}")

    dataset_name = (
        "princeton-nlp/SWE-bench_Lite_bm25_13K"
        if disable_cognee
        else "princeton-nlp/SWE-bench_Lite"
    )

    swe_dataset = load_swebench_dataset(dataset_name, split="test")
    swe_dataset = swe_dataset[:num_samples]

    tasks = [
        run_in_docker(instance, idx, disable_cognee, dockerfile_path=dockerfile_path)
        for idx, instance in enumerate(swe_dataset)
    ]

    # Run all containers concurrently and collect results
    results = await asyncio.gather(*tasks)

    # Filter out None values (failed instances)
    successful_results = [res for res in results if res is not None]

    print(f"\nAll {num_samples} samples processed. Results:")
    for idx, result in enumerate(successful_results):
        print(f"Instance {idx + 1}: {result}")

    # Save results to a JSON file
    with open("processed_results.json", "w") as f:
        json.dump(successful_results, f, indent=2)

    print("Results saved to processed_results.json.")


if __name__ == "__main__":
    asyncio.run(main(disable_cognee=True, num_samples=5))
