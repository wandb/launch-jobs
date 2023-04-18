"""
This job dispatches a GitHub Actions workflow.

Inputs (configs):
- owner,repo,ref,workflow,workflow_inputs: https://docs.github.com/en/rest/actions/workflows?apiVersion=2022-11-28#create-a-workflow-dispatch-event
- github_api_token_env_var: The name of the environment variable that contains the GitHub API token.
- retry_settings: The retry settings for the HTTP request.

Outputs:
- None
"""

import os
from pathlib import Path

import yaml
from github import Github
from tenacity import Retrying, stop_after_attempt, wait_random_exponential

import wandb

# Used to load example configs from wandb jobs repo.
# Is there a better way to handle this?
p = Path("config.yml")
if p.is_file():
    with open(p) as f:
        config = yaml.safe_load(f)

with wandb.init(config=config, job_type="webhook") as run:
    token = os.getenv(run.config.github_api_token_env_var)
    workflow = (
        Github(token)
        .get_user(run.config.owner)
        .get_repo(run.config.repo)
        .get_workflow(run.config.workflow)
    )
    for attempt in Retrying(
        stop=stop_after_attempt(run.config.retry_settings["attempts"]),
        wait=wait_random_exponential(**run.config.retry_settings["backoff"]),
    ):
        with attempt:
            workflow.create_dispatch(run.config.ref, run.config.workflow_inputs)
