import os

import polars as pl
import yaml

import wandb

# Used to load example configs from wandb jobs repo.
# Is there a better way to handle this?
config = {}
cfg = os.getenv("WANDB_JOBS_REPO_CONFIG")
if cfg:
    with open(cfg) as f:
        config = yaml.safe_load(f)


with wandb.init(config=config, job_type="data_pipeline") as run:
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
