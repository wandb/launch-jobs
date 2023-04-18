import os
from pathlib import Path

import polars as pl
import yaml

import wandb

# Used to load example configs from wandb jobs repo.
# Is there a better way to handle this?
p = Path("config.yml")
if p.is_file():
    with open(p) as f:
        config = yaml.safe_load(f)


with wandb.init(config=config, job_type="data_pipeline") as run:
    protocol = run.config.connection["protocol"]
    base_url = run.config.connection["base_url"]
    username = os.getenv(run.config.connection["username_env"])
    password = os.getenv(run.config.connection["password_env"])
    conn = f"{protocol}://{username}:{password}@{base_url}"

    df = pl.read_sql(run.config.query, conn)
    run.log({run.config.table_name: df.to_pandas()})
