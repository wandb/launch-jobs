import argparse
import os

import click
import wandb
from wandb import termlog
from wandb.apis.internal import Api
from wandb.sdk.launch.sweeps.scheduler import Scheduler


def setup_scheduler(scheduler: Scheduler, **kwargs):
    """Setup a run to log a scheduler job.

    If this job is triggered using a sweep config, it will
    become a sweep scheduler, automatically managing a launch sweep
    Otherwise, we just log the code, creating a job that can be
    inserted into a sweep config."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default=kwargs.get("project"))
    parser.add_argument("--entity", type=str, default=kwargs.get("entity"))
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--enable_git", action="store_true", default=False)
    cli_args = parser.parse_args()

    name = cli_args.name or scheduler.__name__
    run = wandb.init(
        settings={"disable_git": True} if not cli_args.enable_git else {},
        project=cli_args.project,
        entity=cli_args.entity,
    )
    config = run.config

    if not config.get("sweep_args", {}).get("sweep_id"):
        _handle_job_logic(run, name, cli_args.enable_git)
        return

    args = config.get("sweep_args", {})
    if cli_args.num_workers:  # override
        kwargs.update({"num_workers": cli_args.num_workers})

    _scheduler = scheduler(Api(), run=run, **args, **kwargs)
    _scheduler.start()


def _handle_job_logic(run, name, enable_git=False) -> None:
    termlog("\nJob not configured to run a sweep, logging code and returning early.")
    jobstr = f"{run.entity}/{run.project}/job"

    if os.environ.get("WANDB_DOCKER"):
        termlog("Identified 'WANDB_DOCKER' environment var, creating image job...")
        tag = os.environ.get("WANDB_DOCKER", "").split(":")
        if len(tag) == 2:
            jobstr += f"-{tag[0].replace('/', '_')}_{tag[-1]}:latest"
        else:
            jobstr = f"found here: https://wandb.ai/{jobstr}s/"
        termlog(f"Creating image job {click.style(jobstr, fg='yellow')}\n")
    elif not enable_git:
        jobstr += f"-{name}:latest"
        termlog(f"Creating code-artifact job: {click.style(jobstr, fg='yellow')}\n")
    else:
        _s = click.style(f"https://wandb.ai/{jobstr}s/", fg="yellow")
        termlog(f"Creating repo-artifact job found here: {_s}\n")
        run.log_code(name=name, exclude_fn=lambda x: x.startswith("_"))
    return
