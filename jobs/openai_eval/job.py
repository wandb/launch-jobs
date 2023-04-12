import datetime
import json
import os
import subprocess

import openai
import pandas as pd
import wandb.apis.reports as wr
from pathlib import Path
import yaml

import wandb


def expand(df, col):
    return pd.concat([df, pd.json_normalize(df[col])], axis=1).drop(col, axis=1)


def extend_chatml(r):
    extensions = [
        # {'role': 'options', 'content': r.options},
        {"role": "assistant", "content": r.sampled, "correct": r.correct}
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


openai.api_key = os.getenv("OPENAI_API_KEY")

p = Path("config.yml")
if p.is_file():
    with open(p) as f:
        config = yaml.safe_load(f)

with wandb.init(config=config, settings=wandb.Settings(disable_git=True)) as run:
    settings = run.config
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

    spec = result.iloc[0, 0]
    final_report = result.iloc[1, 1]

    # There is probably a better way to do this, but I can't seem to capture the fname...
    new_fname = f"{spec['run_id']}_{settings['model']}_{settings['eval']}.jsonl"
    os.rename(record_path, new_fname)

    df = result.iloc[2:, 2:].pipe(expand, "data")

    key = ["run_id", "event_id", "sample_id"]
    sampling_cols = ["prompt", "created_at"]
    match_cols = ["sampled", "correct", "expected", "picked", "options"]

    df_sampling = df.loc[df.type == "sampling", key + sampling_cols].reset_index(
        drop=True
    )
    df_match = (
        df.loc[df.type == "match", key + match_cols]
        .assign(event_id=lambda df: df.event_id - 1)
        .reset_index(drop=True)
    )
    df_merged = df_sampling.merge(df_match).drop("event_id", axis=1)
    df_final = df_merged.assign(
        extended_convo=df_merged.apply(extend_chatml, axis=1)
    ).pipe(
        lambda df: df.assign(
            model=settings["model"],
            chatml_viz=df.extended_convo.apply(make_chatml_viz),
            prompt=df.prompt.apply(json.dumps),
            extended_convo=df.extended_convo.apply(json.dumps),
        )
    )
    run.log(
        {
            "final_report": final_report,
            "sampling": df_final,
            "spec": spec,
        }
    )

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
