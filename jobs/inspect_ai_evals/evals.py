import os

import wandb
import weave
from weave.trace.context.weave_client_context import set_weave_client_global
from inspect_ai import eval_set
from inspect_ai._eval.loader import load_tasks
from inspect_ai._eval.task import task_with
from inspect_ai.model import get_model
from wandb.sdk import launch

from leaderboard import create_leaderboard
from exception import handle_exception

def main():
    with wandb.init(config=launch.load_wandb_config()) as run:
        weave_client = weave.init(run.project)

        hf_token = run.config.get("hf_token")
        if hf_token:
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
            os.environ.setdefault("HF_TOKEN", hf_token)
            os.environ.setdefault("HUGGINGFACE_TOKEN", hf_token)

        api_key = run.config["model"].get("api_key_var")
        if api_key:
            os.environ.setdefault("OPENAI_API_KEY", api_key)

        model = get_model(
            run.config["model"]["model_name"],
            base_url=run.config["model"].get("base_url"),
            api_key=api_key,
        )
        tasks = [
            task_with(load_tasks([task])[0], model=model)
            for task in run.config["tasks"]
        ]

        try:
            success, _ = eval_set(
                tasks=tasks,
                log_dir="logs/",
                limit=run.config.get("limit", 5),
                log_dir_allow_dirty=True,
            )
        except Exception as e:
            handle_exception(e)
            
        if not success:
            raise Exception("Evaluation set did not run successfully.")

        if run.config.get("create_leaderboard", True):
            create_leaderboard()
        
        weave_client.finish()


if __name__ == "__main__":
    main()
