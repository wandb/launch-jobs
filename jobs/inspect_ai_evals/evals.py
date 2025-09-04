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

def main():
    with wandb.init(config=launch.load_wandb_config()) as run:
        weave_client = weave.init(f"{run.entity}/{run.project}")

        hf_token = get_launch_secret_from_env("hf_token", run.config)
        if hf_token:
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
            os.environ.setdefault("HF_TOKEN", hf_token)
            os.environ.setdefault("HUGGINGFACE_TOKEN", hf_token)

        api_key = get_launch_secret_from_env("api_key_var", run.config["model"])
        if api_key:
            os.environ.setdefault("OPENAI_API_KEY", api_key)

        model_name = run.config["model"]["model_name"]
        try:
            model = get_model(
                model_name,
                base_url=run.config["model"].get("base_url"),
                api_key=api_key,
            )
                
            os.environ.setdefault("INSPECT_EVAL_MODEL", model_name)
            os.environ.setdefault("INSPECT_GRADER_MODEL", model_name)
        except Exception as e:
            wandb.termerror(f"Error initializing model. Please check if the base URL is valid and the API key is correct: {e}")
            weave_client.finish()
            raise e

        
        failed_tasks = []
        for task in run.config["tasks"]:
            try:
                loaded_task = [task_with(load_tasks([task])[0], model=model)]
                success, _ = eval_set(
                    tasks=loaded_task,
                    log_dir="logs/",
                    limit=run.config.get("limit", 5),
                    log_dir_allow_dirty=True,
                )
                
                if not success:
                    wandb.termerror(f"Task {task} did not run successfully")
                    failed_tasks.append(task)
                    continue
                
                if run.config.get("create_leaderboard", True):
                    create_leaderboard()

            except Exception:
                wandb.termerror(f"Task {task} failed to run")
                failed_tasks.append(task)
                
        if failed_tasks:
            wandb.termerror(f"Failed to run tasks: {failed_tasks}")
            
        weave_client.finish()


if __name__ == "__main__":
    main()
