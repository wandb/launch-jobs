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
from datasets.exceptions import DatasetNotFoundError

INSPECT_EVAL_PREFIX = "inspect_evals/"


def main():
    config = launch.load_wandb_config()
    with wandb.init(config=dict(config)) as run:
        weave_client = weave.init(f"{run.entity}/{run.project}")

        _, hf_token = get_launch_secret_from_env("hf_token", run.config)
        if hf_token:
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
            os.environ.setdefault("HF_TOKEN", hf_token)
            os.environ.setdefault("HUGGINGFACE_TOKEN", hf_token)

        api_key_name, api_key = get_launch_secret_from_env(
            "api_key_var", run.config.get("model", {})
        )
        if api_key_name and api_key:
            os.environ.setdefault(api_key_name, api_key)

        # Some evals use an OpenAI model as the default scorer.
        # Otherwise, we can use the model selected by the user (see INSPECT_GRADER_MODEL)
        _, scorer_api_key = get_launch_secret_from_env("scorer_api_key", run.config)
        if scorer_api_key:
            os.environ.setdefault("OPENAI_API_KEY", scorer_api_key)

        try:
            model_name = run.config.get("model", {}).get("model_name")
            base_url = run.config.get("model", {}).get("base_url", None)
            if not model_name:
                wandb.termerror("Hint: Model name is required")
                weave_client.finish()
                return

            model = get_model(model_name, api_key=api_key, base_url=base_url)

            os.environ.setdefault("INSPECT_EVAL_MODEL", model_name)
            os.environ.setdefault("INSPECT_GRADER_MODEL", model_name)
        except Exception as e:
            wandb.termerror(f"Error initializing model: {e}")
            wandb.termlog(
                "Hint: Please check if the job inputs for the model is correct ('Name' and 'API Key')"
            )
            weave_client.finish()
            raise e

        failed_tasks = []
        for task in run.config.get("tasks", []):
            try:
                loaded_task = [
                    task_with(
                        load_tasks([f"{INSPECT_EVAL_PREFIX}{task}"])[0], model=model
                    )
                ]
                success, _ = eval_set(
                    tasks=loaded_task,
                    log_dir="logs/",
                    limit=run.config.get("limit", None),
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
                if isinstance(e, DatasetNotFoundError):
                    wandb.termlog(
                        "Hint: This may be a gated dataset. Please check that you have set the 'Hugging Face Token' in the job input and have accepted the agreement on Hugging Face."
                    )
            run.finish(exit_code=1)

        weave_client.finish()


if __name__ == "__main__":
    main()
