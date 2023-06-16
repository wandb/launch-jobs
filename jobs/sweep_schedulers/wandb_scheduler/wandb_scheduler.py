"""Scheduler for classic wandb Sweeps."""
import os
import logging
import argparse
import click
from pprint import pformat as pf
from typing import Any, Dict, List, Optional

import wandb
from wandb.apis.internal import Api
from wandb.sdk.launch.sweeps.scheduler import LOG_PREFIX, RunState, Scheduler, SweepRun


_logger = logging.getLogger(__name__)


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
    wandb.termlog(
        "\nJob not configured to run a sweep, logging code and returning early."
    )
    jobstr = f"{run.entity}/{run.project}/job"

    if os.environ.get("WANDB_DOCKER"):
        wandb.termlog(
            "Identified 'WANDB_DOCKER' environment var, creating image job..."
        )
        tag = os.environ.get("WANDB_DOCKER", "").split(":")
        if len(tag) == 2:
            jobstr += f"-{tag[0].replace('/', '_')}_{tag[-1]}:latest"
        else:
            jobstr = f"found here: https://wandb.ai/{jobstr}s/"
        wandb.termlog(f"Creating image job {click.style(jobstr, fg='yellow')}\n")
    elif not enable_git:
        jobstr += f"-{name}:latest"
        wandb.termlog(
            f"Creating code-artifact job: {click.style(jobstr, fg='yellow')}\n"
        )
    else:
        _s = click.style(f"https://wandb.ai/{jobstr}s/", fg="yellow")
        wandb.termlog(f"Creating repo-artifact job found here: {_s}\n")
        run.log_code(name=name, exclude_fn=lambda x: x.startswith("_"))
    return


class WandbScheduler(Scheduler):
    """A controller/agent that populates a Launch RunQueue from a sweeps RunQueue."""

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

    def _get_next_sweep_run(self, worker_id: int) -> Optional[SweepRun]:
        """Called by the main scheduler execution loop.

        Expected to return a properly formatted SweepRun if the scheduler
        is alive, or None and set the appropriate scheduler state:

        FAILED: self.fail_sweep()
        STOPPED: self.stop_sweep()
        """
        commands: List[Dict[str, Any]] = self._get_sweep_commands(worker_id)
        for command in commands:
            # The command "type" can be one of "run", "resume", "stop", "exit"
            _type = command.get("type")
            if _type in ["exit", "stop"]:
                self.stop_sweep()
                return None

            if _type not in ["run", "resume"]:
                self.fail_sweep(f"AgentHeartbeat unknown command: {_type}")

            _run_id: Optional[str] = command.get("run_id")
            if not _run_id:
                self.fail_sweep(f"No run id in agent heartbeat: {command}")
                return None

            if _run_id in self._runs:
                wandb.termlog(f"{LOG_PREFIX}Skipping duplicate run: {_run_id}")
                continue

            return SweepRun(
                id=_run_id,
                state=RunState.PENDING,
                args=command.get("args", {}),
                logs=command.get("logs", []),
                worker_id=worker_id,
            )
        return None

    def _get_sweep_commands(self, worker_id: int) -> List[Dict[str, Any]]:
        """Helper to recieve sweep command from backend."""
        # AgentHeartbeat wants a Dict of runs which are running or queued
        _run_states: Dict[str, bool] = {}
        for run_id, run in self._yield_runs():
            # Filter out runs that are from a different worker thread
            if run.worker_id == worker_id and run.state.is_alive:
                _run_states[run_id] = True

        _logger.debug(f"Sending states: \n{pf(_run_states)}\n")
        commands: List[Dict[str, Any]] = self._api.agent_heartbeat(
            agent_id=self._workers[worker_id].agent_id,
            metrics={},
            run_states=_run_states,
        )
        _logger.debug(f"AgentHeartbeat commands: \n{pf(commands)}\n")

        return commands

    def _exit(self) -> None:
        pass

    def _poll(self) -> None:
        _logger.debug(f"_poll. _runs: {self._runs}")
        pass

    def _load_state(self) -> None:
        pass

    def _save_state(self) -> None:
        pass


if __name__ == "__main__":
    setup_scheduler(WandbScheduler)
