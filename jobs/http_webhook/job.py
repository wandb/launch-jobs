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
from tenacity import Retrying, stop_after_attempt, wait_random_exponential

import wandb

settings = wandb.Settings(disable_git=True)

with wandb.init(settings=settings) as run:
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
