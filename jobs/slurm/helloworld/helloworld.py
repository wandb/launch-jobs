import wandb
from wandb.sdk import launch
import os
import time

name = f"{os.environ["SLURM_JOB_NAME"]} ({os.environ["SLURM_JOB_ID"]})"
wandb.init(project="helloworld-slurm", name=name, group=f"slurm-{os.environ["SLURM_JOB_ID"]}", config={"hello": "default"})
launch.manage_wandb_config(include=["hello"])

print("wandb.config: ", dict(wandb.config))
time.sleep(30)
print("Goodbye world!")