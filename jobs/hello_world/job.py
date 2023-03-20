import os

import yaml

import wandb

# Used to load example configs from wandb jobs repo.
# Is there a better way to handle this?
cfg = os.getenv("WANDB_JOBS_REPO_CONFIG")
if cfg:
    with open(cfg) as f:
        config = yaml.safe_load(f)

with wandb.init(config=config) as run:
    run.log({"hello": "world"})
