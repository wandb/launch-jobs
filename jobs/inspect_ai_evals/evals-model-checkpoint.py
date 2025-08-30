import os

import wandb
import weave
from inspect_ai import eval_set
from inspect_ai._eval.loader import load_tasks
from inspect_ai._eval.task import task_with
from inspect_ai.model import get_model
from wandb.sdk import launch

from leaderboard import create_leaderboard
from exception import handle_exception, UnsuccessfulEvaluation
from launch_secrets import get_launch_secret_from_env


import time
import requests

MAX_TIMEOUT = int(os.environ.get("VLLM_STARTUP_TIMEOUT", "900"))  # default 15 minutes
API_KEY = "token-abc123"


def wait_for_vllm(
    api_key: str,
    max_timeout: int,
    run: wandb.sdk.wandb_run.Run,
) -> str:
    """
    Probe candidate URLs and wait until the vLLM server becomes healthy.

    Tries multiple endpoints commonly exposed by vLLM: /health, /v1/models, and /.

    Args:
        api_key (str): Bearer token used for Authorization.
        max_timeout (int): Maximum time to wait in seconds.
        run (wandb.sdk.wandb_run.Run): The active W&B run.

    Raises:
        TimeoutError: If the server is not healthy within the allowed time.
    """
    start_time = time.time()
    base_url = (
        f"http://evals-{run.entity}-{run.project}-{run.id}.wandb.svc.cluster.local:8000"
    )
    while time.time() - start_time < max_timeout:
        try:
            resp = requests.get(
                f"{base_url}/health",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=3.0,
            )
            if 200 <= resp.status_code < 300:
                return base_url
        except Exception:
            pass
        time.sleep(5)

    raise TimeoutError(f"VLLM server failed to start within {max_timeout} seconds")


def main():
    with wandb.init(config=launch.load_wandb_config()) as run:
        print("Waiting for VLLM server to start...")
        server_base = wait_for_vllm(API_KEY, MAX_TIMEOUT, run)
        print(f"VLLM server started at {server_base}")

        openai_api_key = get_launch_secret_from_env("openai_api_key", run.config)
        os.environ.setdefault("OPENAI_API_KEY", openai_api_key or API_KEY)
        os.environ.setdefault("AZURE_OPENAI_API_KEY", openai_api_key or API_KEY)

        hf_token = get_launch_secret_from_env("hf_token", run.config)
        if hf_token:
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
            os.environ.setdefault("HF_TOKEN", hf_token)
            os.environ.setdefault("HUGGINGFACE_TOKEN", hf_token)

        weave_client = weave.init(f"{run.entity}/{run.project}")

        try:
            model = get_model(
                "vllm/user-model",
                base_url=f"{server_base}/v1",
                api_key=API_KEY,
            )
            tasks = [
                task_with(load_tasks([task])[0], model=model)
                for task in run.config["tasks"]
            ]

            success, _ = eval_set(
                tasks=tasks,
                log_dir="logs/",
                limit=run.config.get("limit", 5),
                log_dir_allow_dirty=True,
            )

            if not success:
                raise UnsuccessfulEvaluation()

            if run.config.get("create_leaderboard", True):
                create_leaderboard()

        except Exception as e:
            handle_exception(e)
            raise e
        finally:
            weave_client.finish()


if __name__ == "__main__":
    main()
