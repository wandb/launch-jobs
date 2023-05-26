import argparse

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
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--disable_git", type=bool, default=True)
    cli_args = parser.parse_args()

    name = cli_args.name or scheduler.__name__
    run = wandb.init(
        settings={"disable_git": True} if cli_args.disable_git else {},
        project=cli_args.project,
        entity=cli_args.entity,
    )
    config = run.config

    if not config.get("sweep_args", {}).get("sweep_id"):
        termlog("Job not configured to run a sweep, logging code and returning early.")
        run.log_code(name=name, exclude_fn=lambda x: x.startswith("_"))
        if cli_args.disable_git:  # too hard to figure out git repo job names
            jobstr = f"{run.entity}/{run.project}/job-{name}:latest"
            termlog(f"Creating job: {click.style(jobstr, fg='yellow')}")
        return

    args = config.get("sweep_args", {})
    if cli_args.num_workers:  # override
        kwargs.update({"num_workers": cli_args.num_workers})

    _scheduler = scheduler(Api(), run=run, **args, **kwargs)
    _scheduler.start()
