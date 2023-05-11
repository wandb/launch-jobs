import argparse

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
    cli_args = parser.parse_args()

    name = cli_args.name or f"{scheduler.__name__}-scheduler-job"

    run = wandb.init(project=cli_args.project, entity=cli_args.entity)
    run.log_code(name=name, exclude_fn=lambda x: x.startswith("_"))
    config = run.config

    if not config.get("sweep_args", {}).get("sweep_id"):
        termlog("Job not configured to run a sweep, logging code and returning early.")
        return

    args = config.get("sweep_args", {})
    wandb.termlog(f"Starting sweep scheduler with args: {args}")

    num_workers = kwargs.pop("num_workers", None)
    if cli_args.num_workers:
        num_workers = cli_args.num_workers

    _scheduler = scheduler(
        Api(), **args, **kwargs, num_workers=num_workers
    )
    _scheduler.start()
