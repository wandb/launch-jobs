import datetime
import json
import os
import shutil
import subprocess
from functools import reduce
from pathlib import Path

import openai
import pandas as pd
import wandb.apis.reports as wr
import yaml

import wandb


def expand(df, col):
    return pd.concat(
        [
            df.reset_index(drop=True),
            pd.json_normalize(df[col]).reset_index(drop=True),
        ],
        axis=1,
    ).drop(col, axis=1)


def extend_chatml(r):
    def get_correct(r):
        if hasattr(r, "correct"):
            return r.correct
        if hasattr(r, "choice"):
            return True if r.choice == "Y" else False
        return None

    extensions = [
        # {'role': 'options', 'content': r.options},
        {"role": "assistant", "content": r.sampled, "correct": get_correct(r)}
    ]
    try:
        return r.prompt + extensions
    except:
        return []


def make_chatml_viz(convo):
    styles = """
    <style>
        /* Message Bubble Container */
        .message-bubble {
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: flex-start;
        margin-bottom: 10px;
        max-width: 600px;
        }

        /* Role Section */
        .message-role {
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 5px;
        color: #555;
        }

        /* Message Section */
        .message-content {
        background-color: #e6e6e6;
        padding: 10px;
        border-radius: 10px;
        font-size: 16px;
        max-width: 100%;
        word-wrap: break-word;
        box-shadow: 0px 1px 1px rgba(0, 0, 0, 0.2);
        }

        /* System messages */
        .message-bubble.system .message-content {
        background-color: #f2f2f2;
        color: #999;
        }

        /* Assistant messages */
        .message-bubble.assistant .message-role {
        color: #6666ff;
        }

        .message-bubble.assistant .message-content {
        background-color: #e6e6ff;
        color: #000;
        }

        /* Sampled messages */
        .message-bubble.sampled {
        justify-content: flex-end;
        align-items: flex-end;
        margin-left: 20%;
        }

        .message-bubble.sampled .message-role {
        color: #1e6ba1;
        text-align: right;
        }

        .message-bubble.sampled .message-content {
        background-image: linear-gradient(to left, #1e6ba1, #7fb1dd);
        color: white;
        /* fallback for browsers that don't support gradients */
        background-color: #1e6ba1;
        /* adjust these values to control the gradient effect */
        background-size: 150% 100%;
        background-position: right;
        transition: background-position 0.3s ease-out;
        border-radius: 10px;
        padding: 10px;
        max-width: 80%;
        word-wrap: break-word;
        box-shadow: 0px 1px 1px rgba(0, 0, 0, 0.2);
        }

        /* Picked messages */
        .message-bubble.picked .message-role {
        color: #555;
        }

        .message-bubble.assistant.true .message-content {
        background-color: #c3e6cb;
        color: #000;
        }

        .message-bubble.assistant.false .message-content {
        background-color: #f5c6cb;
        color: #000;
        }
        
        .message-bubble.assistant.true .message-role {
        color: #006400;
        }

        .message-bubble.assistant.false .message-role {
        color: #8b0000;
        }

        /* Right-aligned message bubble for the user */
        .message-bubble.user {
        justify-content: flex-end;
        align-items: flex-end;
        margin-left: 20%;
        }

        /* Role section for the user */
        .message-bubble.user .message-role {
        color: #555;
        text-align: right;
        }

        /* Message section for the user */
        .message-bubble.user .message-content {
        background-image: linear-gradient(to right, #80b6f4, #5c7cfa);
        color: white;
        /* fallback for browsers that don't support gradients */
        background-color: #5c7cfa;
        /* adjust these values to control the gradient effect */
        background-size: 150% auto;
        background-position: left center;
        text-shadow: 0px 0px 2px rgba(0, 0, 0, 0.2);
        box-shadow: none;
        }
    </style>
    """

    msgs = "\n".join(stylize_msg(msg) for msg in convo)
    html = f"{styles}{msgs}"

    return wandb.Html(html)


def stylize_msg(msg):
    correct = msg.get("correct")
    if correct is True:
        correct = "true"
    if correct is False:
        correct = "false"
    return f"""
    <div class="message-bubble {msg['role']} {correct}">
        <div class="message-role">{msg['role']}</div>
        <div class="message-content">{msg['content']}</div>
    </div>
    """


def chatml_to_markdown(convo):
    md = ""
    for msg in convo:
        role = msg["role"]
        content = msg["content"]
        # correct = msg.get("correct")
        # content = f'<span style="color:blue">{content}</span>'
        md += f"# {role}\n{content}\n\n"

    return md


def get_expand_subsets(df, col):
    return {
        subset: df.loc[df[col] == subset].pipe(expand, "data").reset_index(drop=True)
        for subset in df[col].unique()
    }


def run_eval(settings):
    if any(k not in settings for k in ["model", "eval"]):
        raise ValueError("`model` and `eval` must be specified in `oaieval_settings`")
    if "record_path" in settings:
        del settings["record_path"]
        wandb.termwarn("Using `record_path` in `oaieval_settings` is not supported.")

    if custom_registry := run.config.get("custom_registry"):
        art_path = custom_registry.download()
        shutil.copytree(art_path, "/setup/evals/evals", dirs_exist_ok=True)

    kwarg_settings = {k: v for k, v in settings.items() if k not in ["model", "eval"]}
    args = [settings["model"], settings["eval"]]
    for k, v in kwarg_settings.items():
        args.append(f"--{k}={v}")

    record_path = settings.get("record_path", "temp.jsonl")
    cmd = ["oaieval"] + args + ["--record_path", record_path]
    subprocess.run(cmd, check=True, timeout=600)


def get_evals_table(test_results):
    df = test_results.iloc[2:, 2:]
    dfs = get_expand_subsets(df, "type")

    key = {"run_id", "sample_id"}
    overlapping_cols = reduce(lambda a, b: set(a) & set(b), dfs.values())

    df2 = dfs["sampling"]
    eval_levels = ["", "modelgraded_", "meta_modelgraded_"]
    eval_dfs = []
    for i, lvl in enumerate(eval_levels):
        df3 = (
            df2.groupby(list(key))
            .nth(i)
            .pipe(lambda df: df.assign(extended_convo=df.apply(extend_chatml, axis=1)))
            .pipe(
                lambda df: df.assign(
                    chatml_viz=df.extended_convo.apply(make_chatml_viz),
                    markdown=df.extended_convo.apply(chatml_to_markdown),
                    prompt=df.prompt.apply(json.dumps),
                    extended_convo=df.extended_convo.apply(json.dumps),
                )
            )
            .add_prefix(lvl)
        )
        if len(df3) > 0:
            eval_dfs.append(df3)

    eval_df = reduce(lambda x, y: x.join(y), eval_dfs).reset_index()
    dfs["sampling"] = eval_df

    dfs2 = {}
    for k, _df in dfs.items():
        extra_drop = set(("created_at", "sampled")) if k == "sampling" else set()
        drops = overlapping_cols - key - extra_drop
        dfs2[k] = _df.drop(columns=drops).reset_index(drop=True)

    final_df = reduce(
        lambda df, df2: df.merge(df2, on=list(key)), dfs2.values()
    ).assign(
        completion_fns=str(spec["completion_fns"]),
        eval_name=spec["eval_name"],
    )

    scary_cols = final_df.columns[
        (final_df.dtypes == "object") & ~final_df.columns.str.contains("chatml_viz")
    ]
    for col in scary_cols:
        final_df[col] = final_df[col].astype(str)

    return final_df


def generate_report(run, settings):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    models = ["gpt-4", "gpt-3.5-turbo"]

    report = wr.Report(
        entity=run.entity,
        project=run.project,
        title=f"{settings['model']}: {settings['eval']}",
        description=f"{now}",
        width="fluid",
        blocks=[
            wr.H1("Sampling Summary"),
            wr.PanelGrid(
                panels=[
                    wr.WeavePanelSummaryTable("sampling", layout={"w": 24, "h": 16})
                ],
                runsets=[
                    wr.Runset(
                        run.entity, run.project, name=model
                    ).set_filters_with_python_expr(f"model == '{model}'")
                    for model in models
                ],
            ),
        ],
    ).save()


def get_test_results():
    return pd.read_json("temp.jsonl", lines=True)


def get_spec(test_results):
    return test_results.iloc[0, 0]


def get_final_report(test_results):
    return test_results.iloc[1, 1]


openai.api_key = os.getenv("OPENAI_API_KEY")

p = Path(os.getenv("_WANDB_CONFIG_FILENAME"))
if p.is_file():
    with p.open() as f:
        config = yaml.safe_load(f)

with wandb.init(config=config, settings=wandb.Settings(disable_git=True)) as run:
    settings = run.config["eval_settings"]
    run_eval(settings)

    test_results = get_test_results()
    spec = get_spec(test_results)
    final_report = get_final_report(test_results)
    evals = get_evals_table(test_results)

    run.log({"evals": evals, **final_report})
    art = wandb.Artifact("results", type="results")
    art.add_file("temp.jsonl")
    run.log_artifact(art)

    generate_report(run, settings)
