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
    filename = "{name}.{filetype}".format(**run.config.artifact)
    df.write_ipc(filename)

    art = wandb.Artifact(run.config.artifact["name"], type="sql-table")
    art.add_file(filename)

    run.log_artifact(art)
