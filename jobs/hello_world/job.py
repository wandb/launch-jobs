import os
from pathlib import Path

import yaml

import wandb

# Used to load example configs from wandb jobs repo.
# Is there a better way to handle this?
p = Path("config.yml")
if p.is_file():
    with open(p) as f:
        config = yaml.safe_load(f)


with wandb.init(config=config) as run:
    run.log({"hello": "world"})
