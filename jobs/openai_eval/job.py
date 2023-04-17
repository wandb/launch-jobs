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
        md += f"# {role}\n{content}\n\n"

    return md


def get_expand_subsets(df, col):
    return {
        subset: df.loc[df[col] == subset].pipe(expand, "data").reset_index(drop=True)
        for subset in df[col].unique()
    }


def override_base_prompt(convo, new_prompt):
    for i, msg in enumerate(convo):
        if msg.get("role") != "system":
            break

    new_convo = [{"role": "system", "content": new_prompt}] + convo[i:]
    return new_convo


def run_eval(run, settings):
    if any(k not in settings for k in ["model", "eval"]):
        raise ValueError("`model` and `eval` must be specified in `oaieval_settings`")
    if "record_path" in settings:
        del settings["record_path"]
        wandb.termwarn("Using `record_path` in `oaieval_settings` is not supported.")

    if custom_registry := run.config.get("custom_registry"):
        art_path = custom_registry.download()
        shutil.copytree(art_path, "/setup/evals/evals", dirs_exist_ok=True)

    if override_prompt := run.config.get("override_prompt"):
        registry_path = Path("/setup/evals/evals/registry")

        for f in (registry_path / "evals").glob("**/*.yaml"):
            d = yaml.safe_load(f.open())

            if settings["eval"] not in d:
                continue

            eval_id = d[settings["eval"]]["id"]
            jsonl_args = {
                k: v for k, v in d[eval_id]["args"].items() if str(v).endswith(".jsonl")
            }
            break

        for k, v in jsonl_args.items():
            fpath = registry_path / "data" / v
            df = pd.read_json(fpath, lines=True)
            df.input = df.input.apply(override_base_prompt, new_prompt=override_prompt)
            df.to_json(fpath, lines=True, orient="records")

    kwarg_settings = {k: v for k, v in settings.items() if k not in ["model", "eval"]}
    args = [settings["model"], settings["eval"]]
    for k, v in kwarg_settings.items():
        args.append(f"--{k}={v}")

    record_path = settings.get("record_path", "temp.jsonl")
    cmd = ["oaieval"] + args + ["--record_path", record_path]
    subprocess.run(cmd, check=True, timeout=600)


def get_overlapping_cols(dfs):
    cols = {}
    for i, df in enumerate(dfs):
        for col in df:
            if col not in cols:
                cols[col] = [i]
            else:
                cols[col].append(i)

    overlaps = set(col for col, indices in cols.items() if len(set(indices)) > 1)
    return overlaps


def drop_if_exists(df, col):
    if col in df:
        df = df.drop(col, axis=1)
    return df


def merge(dfs, key, drop_duplicates=True, suffixes=("", "__extraneous")):
    df = reduce(lambda x, y: x.merge(y, on=list(key), suffixes=suffixes), dfs)
    if drop_duplicates:
        df = df.drop(df.filter(regex=suffixes[-1]), axis=1)
    return df


def reshape_by_eval_level(df, key):
    eval_levels = ["", "modelgraded_", "meta_modelgraded_"]
    eval_dfs = []
    for i, lvl in enumerate(eval_levels):
        df2 = (
            df.groupby(list(key))
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
        if len(df2) > 0:
            eval_dfs.append(df2)

    eval_df = reduce(lambda x, y: x.join(y), eval_dfs).reset_index()
    return eval_df


def get_evals_table(test_results):
    key = {"sample_id"}

    df = test_results.iloc[2:, 2:]
    dfs = get_expand_subsets(df, "type")
    dfs["sampling"] = dfs["sampling"].pipe(reshape_by_eval_level, key=key)

    final_df = merge(dfs.values(), key).assign(
        completion_fns=str(spec.get("completion_fns")),
        eval_name=spec.get("eval_name"),
    )
    final_df["completion_cost"] = final_df.usage_total_tokens.apply(add_completion_cost)

    if "override_prompt" in run.config:
        final_df["custom_prompt"] = run.config["override_prompt"]

    if "custom_registry" in run.config:
        final_df["registry_version"] = run.config["custom_registry"].version

    scary_cols = final_df.columns[
        (final_df.dtypes == "object") & ~final_df.columns.str.contains("chatml_viz")
    ]
    for col in scary_cols:
        final_df[col] = final_df[col].astype(str)

    return final_df


def generate_report(run, settings):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    models = ["gpt-4", "gpt-3.5-turbo"]

    reference_report = wr.Report.from_url(
        "https://wandb.ai/megatruong/openai-eval105/reports/gpt-3-5-turbo-manga-translation-panel--Vmlldzo0MDg2MzM2"
    )

    report = wr.Report(
        entity=run.entity,
        project=run.project,
        title=f"{settings['model']}: {settings['eval']}",
        description=f"{now}",
        width="fluid",
        blocks=reference_report.blocks,
        # wr.H1("Leaderboard and Summary"),
        # panel_grid_template
        # wr.PanelGrid(
        #     panels=[
        #         wr.WeavePanelSummaryTable("sampling", layout={"w": 24, "h": 16})
        #     ],
        #     runsets=[
        #         wr.Runset(
        #             run.entity, run.project, name=model
        #         ).set_filters_with_python_expr(f"model == '{model}'")
        #         for model in models
        #     ],
        # ),
    ).save()


def get_test_results():
    return pd.read_json("temp.jsonl", lines=True)


def get_spec(test_results):
    return test_results.iloc[0, 0]


def get_final_report(test_results):
    return test_results.iloc[1, 1]


def override_base_prompt(convo, new_prompt):
    for i, msg in enumerate(convo):
        if msg.get("role") != "system":
            break

    new_convo = [{"role": "system", "content": new_prompt}] + convo[i:]
    return new_convo


def add_completion_cost(n_tokens, cost_per_1k_tokens=0.06):
    return (n_tokens // 1000 + 1) * cost_per_1k_tokens


openai.api_key = os.getenv("OPENAI_API_KEY")

p = Path(os.getenv("_WANDB_CONFIG_FILENAME"))
if p.is_file():
    with p.open() as f:
        config = yaml.safe_load(f)

with wandb.init(config=config, settings=wandb.Settings(disable_git=True)) as run:
    settings = run.config["eval_settings"]
    run_eval(run, settings)

    test_results = get_test_results()
    spec = get_spec(test_results)
    final_report = get_final_report(test_results)
    if not isinstance(final_report, dict):
        final_report = {"final_report": final_report}
    evals = get_evals_table(test_results)
    total_completion_cost = evals.completion_cost.sum()

    run.log(
        {
            "evals": wandb.plot_table(
                vega_spec_name="megatruong/test",
                data_table=wandb.Table(dataframe=evals),
                fields={
                    "metric": "sacrebleu_sentence_score",
                    "color": "custom_prompt",
                    "xaxis": "registry_version",
                    "hover": "markdown",
                },
            ),
            "total_completion_cost": total_completion_cost,
            **final_report,
        }
    )
    art = wandb.Artifact("results", type="results")
    art.add_file("temp.jsonl")
    run.log_artifact(art)

    generate_report(run, settings)
