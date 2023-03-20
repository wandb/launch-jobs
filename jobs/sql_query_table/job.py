import os

import polars as pl
import yaml

import wandb

# Used to load example configs from wandb jobs repo.
# Is there a better way to handle this?
cfg = os.getenv("WANDB_JOBS_REPO_CONFIG")
if cfg:
    with open(cfg) as f:
        config = yaml.safe_load(f)


with wandb.init(config=config) as run:
    protocol = run.config.connection["protocol"]
    base_url = run.config.connection["base_url"]
    username = os.getenv(run.config.connection["username_env"])
    password = os.getenv(run.config.connection["password_env"])
    conn = f"{protocol}://{username}:{password}@{base_url}"

    df = pl.read_sql(run.config.query, conn)
    run.log({run.config.table_name: df.to_pandas()})
