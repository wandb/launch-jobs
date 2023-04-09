import os
import subprocess

import pandas as pd
import wandb
import yaml
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

config = {}
cfg = os.getenv("WANDB_JOBS_REPO_CONFIG")
if cfg:
    with open(cfg) as f:
        config = yaml.safe_load(f)


def expand(df, col):
    return pd.concat([df, pd.json_normalize(df[col])], axis=1).drop(col, axis=1)


def remove_no_variance_cols(df):
    unique_values = df.astype(str).nunique()
    cols_with_var = unique_values[unique_values > 1].index
    return df.loc[:, cols_with_var]


with wandb.init(config=config, project="openai-eval-testing2") as run:
    settings = config
    if any(k not in settings for k in ["model", "eval"]):
        raise ValueError("`model` and `eval` must be specified in `oaieval_settings`")
    if "record_path" in settings:
        del settings["record_path"]
        wandb.termwarn("")

    kwarg_settings = {k: v for k, v in settings.items() if k not in ["model", "eval"]}
    args = [settings["model"], settings["eval"]]
    for k, v in kwarg_settings.items():
        args.append(f"--{k}={v}")

    record_path = settings.get("record_path", "temp.jsonl")
    cmd = ["oaieval"] + args + ["--record_path", record_path]
    subprocess.run(cmd, check=True)

    result = pd.read_json(record_path, lines=True)

    # There is probably a better way to do this, but I can't seem to capture the fname...
    spec = result.iloc[0, 0]
    final_report = result.iloc[1, 1]
    new_fname = f"{spec['run_id']}_{settings['model']}_{settings['eval']}.jsonl"
    os.rename(record_path, new_fname)

    df = result.iloc[2:, 2:].reset_index(drop=True)
    df2 = (
        df.pipe(lambda df: df.loc[df.type == "sampling"])
        .reset_index(drop=True)
        .pipe(expand, "data")
    )

    run.config["spec"] = spec
    run.log({**final_report, "sampling": df2})
