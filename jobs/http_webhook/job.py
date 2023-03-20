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

import httpx
import yaml
from tenacity import Retrying, stop_after_attempt, wait_random_exponential

import wandb

# Used to load example configs from wandb jobs repo.
# Is there a better way to handle this?
config = {}
cfg = os.getenv("WANDB_JOBS_REPO_CONFIG")
if cfg:
    with open(cfg) as f:
        config = yaml.safe_load(f)


with wandb.init(config=config, job_type="webhook") as run:
    token = os.getenv(run.config.github_api_token_env_var)
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {
        "ref": run.config["ref"],
        "inputs": run.config["payload_inputs"],
    }

    for attempt in Retrying(
        stop=stop_after_attempt(run.config.retry_settings["attempts"]),
        wait=wait_random_exponential(**run.config.retry_settings["backoff"]),
    ):
        with attempt, httpx.Client(base_url="https://api.github.com") as client:
            endpoint = f"/repos/{run.config.repo}/actions/workflows/{run.config.workflow}/dispatches"  # noqa
            r = client.post(endpoint, headers=headers, json=payload)
            r.raise_for_status()

    run.log_code()
