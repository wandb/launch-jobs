import os

import wandb
import weave
from inspect_ai import eval_set
from inspect_ai._eval.loader import load_tasks
from inspect_ai._eval.task import task_with
from inspect_ai.model import get_model
from wandb.sdk import launch

from leaderboard import create_leaderboard
from launch_secrets import get_launch_secret_from_env


import time
import requests
import re

MAX_TIMEOUT = 900  # 15 minutes
VLLM_API_KEY = "token-abc123"  # This should match the API key set on the vLLM server.
INSPECT_EVAL_PREFIX = "inspect_evals/"


def make_k8s_label_safe(value: str) -> str:
    safe = value.replace("_", "-").lower()
    safe = re.sub(r"[^a-z0-9\-]", "", safe)
    safe = re.sub(r"-+", "-", safe)
    safe = safe[:63].strip("-")
    return safe


def wait_for_vllm(
    run: wandb.sdk.wandb_run.Run,
) -> str:
    endpoint = make_k8s_label_safe(f"evals-{run.id}-{run.project}-{run.entity}")
    base_url = f"http://{endpoint}.wandb.svc.cluster.local:8000"
    start_time = time.time()
    print(f"Waiting for VLLM server to start at {base_url}")
    while time.time() - start_time < MAX_TIMEOUT:
        try:
            resp = requests.get(
                f"{base_url}/health",
                headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
                timeout=3.0,
            )
            if 200 <= resp.status_code < 300:
                return base_url
        except Exception:
            pass
        time.sleep(5)

    raise TimeoutError(
        f"VLLM server failed to respond to health checks within {MAX_TIMEOUT} seconds."
        f" Please verify that your model is vLLM compatible and fits within the 86GB memory limit."
    )


def main():
    config = launch.load_wandb_config()
    with wandb.init(config=dict(config)) as run:
        server_base = wait_for_vllm(run)
        print(f"VLLM server started at {server_base}")

        artifact_path = run.config.get("artifact_path")
        if not artifact_path:
            raise ValueError("Artifact path is required")

        VLLM_MODEL_NAME = f"vllm/{artifact_path}"

        os.environ.setdefault("INSPECT_EVAL_MODEL", VLLM_MODEL_NAME)
        os.environ.setdefault("VLLM_API_KEY", VLLM_API_KEY)
        os.environ.setdefault("VLLM_BASE_URL", f"{server_base}/v1")

        _, scorer_api_key = get_launch_secret_from_env("scorer_api_key", run.config)
        if scorer_api_key:
            os.environ.setdefault("OPENAI_API_KEY", scorer_api_key)
            os.environ.setdefault("AZURE_OPENAI_API_KEY", scorer_api_key)

        _, hf_token = get_launch_secret_from_env("hf_token", run.config)
        if hf_token:
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
            os.environ.setdefault("HF_TOKEN", hf_token)
            os.environ.setdefault("HUGGINGFACE_TOKEN", hf_token)

        weave_client = weave.init(f"{run.entity}/{run.project}")

        try:
            model = get_model(
                VLLM_MODEL_NAME,
                base_url=f"{server_base}/v1",
                api_key=VLLM_API_KEY,
            )
        except Exception as e:
            wandb.termerror(f"Error initializing model: {e}")
            weave_client.finish()
            raise e

        failed_tasks = []
        for task in run.config["tasks"]:
            try:
                loaded_task = [
                    task_with(
                        load_tasks([f"{INSPECT_EVAL_PREFIX}{task}"])[0], model=model
                    )
                ]
                sample_limit = run.config.get("limit", None)
                success, _ = eval_set(
                    tasks=loaded_task,
                    log_dir="logs/",
                    limit=sample_limit
                    if sample_limit
                    else None,  # evaluate all samples if sample_limit is set to 0
                    retry_attempts=1,
                    retry_wait=10,
                    log_dir_allow_dirty=True,
                )

                if not success:
                    wandb.termerror(f"Task {task} did not run successfully")
                    failed_tasks.append(
                        (
                            task,
                            Exception(
                                "Task did not complete successfully. Check the logs for more details."
                            ),
                        )
                    )
                    continue

                if run.config.get("create_leaderboard", True):
                    create_leaderboard()

            except Exception as e:
                wandb.termerror(f"Task {task} failed to run")
                failed_tasks.append((task, e))

        if failed_tasks:
            wandb.termerror(
                f"The following tasks failed to run: {[task for (task, _) in failed_tasks]}"
            )
            for task, e in failed_tasks:
                wandb.termerror(f"Task {task} failed to run with error: {e}")
            run.finish(exit_code=1)

        weave_client.finish()


if __name__ == "__main__":
    main()
