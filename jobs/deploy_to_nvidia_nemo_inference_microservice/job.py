import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal, Optional

import boto3
import wandb
import yaml
from pydantic import BaseModel
from rich.logging import RichHandler

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
    ],
)
logger = logging.getLogger(__name__)


model_config_mapping = {
    "llama": "llama_template.yaml",
    "starcoder": "starcoder_template.yaml",
}


def run_cmd(cmd: list[str], error_msg: Optional[str] = None, shell: bool = False):
    command = " ".join(cmd) if shell else cmd
    logger.debug(f"Running {command=}")
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        shell=shell,
    ) as proc:
        for line in proc.stdout:
            logger.info(line.strip())

        if proc.poll() != 0:
            logger.error(error_msg)
            sys.exit(1)


class Config(BaseModel):
    artifact: str
    artifact_model_type: Literal["llama", "starcoder"]

    bucket_name: str
    nim_model_store_path: str = "/model-store/"
    s3_model_repo_path: str = "models"

    openai_port: int = 9999
    nemo_port: int = 9998

    deploy_option: Literal[
        "local-nim",
        # "remote-nim",  # in a future release, NIM will have an option to point to external model store
    ] = "local-nim"

    download_artifact: bool = True
    generate_model: bool = True
    update_repo_names: Literal[
        False
    ] = False  # TODO: Add this option back when Nvidia officially supports alt repos
    push_to_s3: bool = False

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


logger.info("Starting deploy to Nvidia Nemo Inference Microservice...")
run = wandb.init()
config = Config(**run.config)

logger.setLevel(config.log_level)
logger.debug(f"{config=}")

artifact_name_cleaned = config.artifact.replace("/", "__").replace(":", "__")
triton_model_name = "ensemble"
triton_trt_model_name = "trt_llm"

base_trt_config_fname = model_config_mapping.get(config.artifact_model_type)
if base_trt_config_fname is None:
    logger.error(f"Unsupported model type {config.artifact_model_type=}, exiting.")
    sys.exit(1)

base_trt_config_path = f"./trt_llm_configs/{base_trt_config_fname}"
with open(base_trt_config_path) as f:
    trt_config = yaml.safe_load(f)


if config.download_artifact:
    logger.info("Downloading model artifact...")
    try:
        art = run.use_artifact(config.artifact)
        artifact_path = art.download()
    except Exception as e:
        logger.error(f"Error downloading artifact, exiting.  {e=}")
        sys.exit(1)


if config.update_repo_names:
    triton_model_name = f"{artifact_name_cleaned}__ensemble"
    triton_trt_model_name = f"{artifact_name_cleaned}__trt_llm"

    trt_config["base_model_id"] = triton_model_name
    trt_config["trt_llm"]["model_name"] = triton_trt_model_name
    trt_config["pipeline"]["model_name"] = triton_model_name


if config.generate_model:
    logger.info("Generating TRT-LLM config from template...")
    trt_config["trt_llm"]["model_path"] = artifact_path

    trt_config_fname = "trt_config.yaml"
    with open(trt_config_fname, "w") as f:
        yaml.dump(trt_config, f)

    logger.info("Running model_repo_generator...")
    cmd = [
        "model_repo_generator",
        "llm",
        "--verbose",
        f"--yaml_config_file={trt_config_fname}",
    ]
    run_cmd(cmd, shell=False)
    logger.info(f"Generated model repos at {config.nim_model_store_path=}")


if config.update_repo_names:
    logger.info("Updating repo to match wandb.Artifact versions...")
    # NOTE: Triton starts at v1, but we start at v0 so we'll be off-by-1.
    # Not sure if this is the best option...
    if config.download_artifact:
        _, ver = art.name.split("v", 1)
        ver = int(ver) + 1
        ver = str(ver)
    else:
        ver = "1"

    base_path = Path(config.nim_model_store_path)
    for model in ["ensemble", "trt_llm"]:
        path = base_path / f"{artifact_name_cleaned}__{model}"
        if path.exists():
            max_ver = max(
                [int(p.name) for p in path.iterdir() if p.is_dir()], default=1
            )
            new_ver = str(max_ver + 1)
            new_path = path / new_ver

            logging.info(f"Adding new model as {new_ver=}")
            src_dir = path / "1"
            shutil.copytree(src_dir, new_path)

# Optional: Push to S3 (in future release, we can load models from here)
if config.push_to_s3:
    logger.info(f"Pushing models to S3 {config.bucket_name=}")
    s3_client = boto3.client("s3")
    for root, _, files in os.walk(config.nim_model_store_path):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, config.nim_model_store_path)
            remote_obj_path = os.path.join(config.s3_model_repo_path, rel_path)
            logger.info(f"Uploading {rel_path} to {remote_obj_path}")
            s3_client.upload_file(full_path, config.bucket_name, remote_obj_path)


if config.deploy_option == "local-nim":
    logger.info("Loading NIM with models locally...")
elif config.deploy_option == "s3-nim":
    ...
    # triton_model_repository = f"s3://{config.bucket_name}/{config.s3_model_repo_path}"

num_gpus = trt_config["trt_llm"]["num_gpus"]
model_name = triton_model_name
openai_port = config.openai_port
nemo_port = config.nemo_port

logger.info("Running inference service...")
cmd = [
    "nemollm_inference_ms",
    f"--{model_name=}",
    f"--{num_gpus=}",
    f"--{openai_port=}",
    f"--{nemo_port=}",
    f"--{triton_model_name=}",
    # f"--{triton_model_repository=}",
]
run_cmd(cmd, shell=True)

run.finish()
