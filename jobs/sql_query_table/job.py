import os

import polars as pl
import wandb

settings = wandb.Settings(disable_git=True)

with wandb.init(settings=settings) as run:
    protocol = run.config.connection["protocol"]
    base_url = run.config.connection["base_url"]
    username = os.getenv(run.config.connection["username_env"])
    password = os.getenv(run.config.connection["password_env"])
    conn = f"{protocol}://{username}:{password}@{base_url}"

    df = pl.read_sql(run.config.query, conn)
    run.log({run.config.table_name: df.to_pandas()})
