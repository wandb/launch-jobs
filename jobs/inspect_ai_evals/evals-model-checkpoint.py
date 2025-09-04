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

MAX_TIMEOUT = 900  # 15 minutes
VLLM_API_KEY = "token-abc123"  # This should match the API key set on the vLLM server.


def wait_for_vllm(
    run: wandb.sdk.wandb_run.Run,
) -> str:
    base_url = (
        f"http://evals-{run.entity}-{run.project}-{run.id}.wandb.svc.cluster.local:8000"
    )
    start_time = time.time()
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

    raise TimeoutError(f"VLLM server failed to start within {MAX_TIMEOUT} seconds")


def main():
    with wandb.init(config=launch.load_wandb_config()) as run:
        print("Waiting for VLLM server to start...")
        server_base = wait_for_vllm(run)
        print(f"VLLM server started at {server_base}")

        scorer_api_key = get_launch_secret_from_env("scorer_api_key", run.config)
        os.environ.setdefault("OPENAI_API_KEY", scorer_api_key or VLLM_API_KEY)
        os.environ.setdefault("AZURE_OPENAI_API_KEY", scorer_api_key or VLLM_API_KEY)

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
                api_key=VLLM_API_KEY,
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
