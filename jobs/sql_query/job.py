import os

import polars as pl
import wandb

settings = wandb.Settings(disable_git=True)

with wandb.init(settings=settings) as run:
    connection = run.config["connection"]

    protocol = connection["protocol"]
    base_url = connection["base_url"]
    username = os.getenv(connection["username_env"])
    password = os.getenv(connection["password_env"])
    conn = f"{protocol}://{username}:{password}@{base_url}"

    df = pl.read_sql(run.config["query"], conn)

    if run.config["output_type"] == "artifact":
        filename = "{name}.{filetype}".format(**run.config["artifact"])
        df.write_ipc(filename)

        art = wandb.Artifact(run.config["artifact"]["name"], type="sql-table")
        art.add_file(filename)

        run.log_artifact(art)

    elif run.config["output_type"] == "table":
        run.log({run.config["table_name"]: df.to_pandas()})
