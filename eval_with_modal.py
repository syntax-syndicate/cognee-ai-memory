# File: eval_with_modal.py
import modal
import os
from pathlib import Path
import sys
import dotenv

volume = modal.Volume.lookup("repopatches")

dotenv.load_dotenv()
app = modal.App("cognee-runner")

# Get the parent directory path
PARENT_DIR = Path(__file__).resolve().parent.parent


MODAL_DOCKERFILE_PATH = Path("Dockerfile.modal")


# Define ignore patterns
IGNORE_PATTERNS = [
    ".venv/**/*",
    "__pycache__",
    "*.pyc",
    ".git",
    ".pytest_cache",
    "*.egg-info",
    "RAW_GIT_REPOS/**/*",
]

# Create image from Modal-specific Dockerfile
image = (
    modal.Image.from_dockerfile(
        path=MODAL_DOCKERFILE_PATH, gpu="T4", force_build=False, ignore=IGNORE_PATTERNS
    )
    .copy_local_file("pyproject.toml", "pyproject.toml")
    .copy_local_file("poetry.lock", "poetry.lock")
    .env({"ENV": os.getenv("ENV"), "LLM_API_KEY": os.getenv("LLM_API_KEY")})
    .poetry_install_from_file(poetry_pyproject_toml="pyproject.toml")
)


@app.function(
    image=image, gpu="T4", concurrency_limit=5, timeout=86400, volumes={"/repo-patches": volume}
)
async def run_single_repo(instance_data: dict, disable_cognee: bool = False):
    import os
    import json
    from process_single_repo import process_repo  # Import the async function directly

    # Process the instance
    result = await process_repo(instance_data, volume=volume, disable_cognee=disable_cognee)

    print(result)

    return None


@app.local_entrypoint()
async def main(disable_cognee: bool = False, num_samples: int = 1):
    import subprocess
    import json
    import os

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

    for dependency in ["transformers", "sentencepiece", "swebench", "python-dotenv"]:
        check_install_package(dependency)

    from swebench.harness.utils import load_swebench_dataset

    print("Configuration:")
    print("• Running in Modal mode")
    print(f"• Disable Cognee: {disable_cognee}")
    print(f"• Number of samples: {num_samples}")

    dataset_name = (
        "princeton-nlp/SWE-bench_Lite_bm25_13K"
        if disable_cognee
        else "princeton-nlp/SWE-bench_Lite"
    )

    swe_dataset = load_swebench_dataset(dataset_name, split="test")
    swe_dataset = swe_dataset[:num_samples]

    print(f"Processing {num_samples} samples from {dataset_name}")
    import pip

    # Install required dependencies
    pip.main(["install", "pydantic>=2.0.0", "pydantic-settings>=2.0.0"])

    tasks = [
        run_single_repo.remote(instance, disable_cognee=disable_cognee) for instance in swe_dataset
    ]
    import asyncio

    # Run all tasks concurrently
    await asyncio.gather(*tasks)

    print("Done!")
