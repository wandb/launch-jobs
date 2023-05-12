import base64
import wandb
from collections import defaultdict
from functools import partial
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_FAIL
# from hyperopt.mongoexp import MongoTrials
import click 
import logging
import time 
from typing import Any, Dict, List, Optional, Tuple

from wandb.sdk.launch.sweeps.scheduler import Scheduler, SweepRun
from wandb.apis.internal import Api
from wandb.apis.public import Api as PublicApi
from wandb.apis.public import Artifact, QueuedRun, Run
from wandb.sdk.launch.sweeps import SchedulerError

from utils import setup_scheduler


logger = logging.getLogger(__name__)

loggers_to_shut_up = [
    "hyperopt.tpe",
    "hyperopt.fmin",
    "hyperopt.pyll.base",
]
for logger in loggers_to_shut_up:
    logging.getLogger(logger).setLevel(logging.ERROR)

LOG_PREFIX = f"{click.style('hyperopt sched:', fg='bright_yellow')} "


class HyperoptScheduler(Scheduler):

    def __init__(
        self,
        api: Api,
        *args: Optional[Any],
        **kwargs: Optional[Any],
    ):
        super().__init__(api, *args, **kwargs)

        self.num_workers = 2  # hyperopt only supports 1 worker

        self.hyperopt_runs = {}
        self._trials = {}
        self._workers: Dict[str, int] = {}

    def _make_search_space(self, sweep_config: Dict[str, Any]) -> Dict[str, Any]:
        """Use a sweep config to create hyperopt search space"""
        config: Dict[str, Dict[str, Any]] = defaultdict(dict)
        for param, extras in sweep_config.items():
            if extras.get("values"):
                config[param] = hp.choice(param, extras["values"])
            elif extras.get("value"):
                config[param] = hp.choice(param, [extras["value"]])
            elif isinstance(extras.get("min"), float):
                if not extras.get("max"):
                    raise SchedulerError(
                        f"{LOG_PREFIX}Error converting config. 'min' requires 'max'"
                    )
                config[param] = hp.uniform(param, extras["min"], extras["max"])
            elif isinstance(extras.get("min"), int):
                if not extras.get("max"):
                    raise SchedulerError(
                        f"{LOG_PREFIX}Error converting config. 'min' requires 'max'"
                    )
                config[param] = hp.randint(param, extras["min"], extras["max"])
            else:
                logger.debug(f"Unknown parameter type: param={param}, val={extras}")
        return config
    
    def _convert_search_space(self, params: Dict[str, Any]) -> Dict[str, Any]:
        wandb_config = {}
        for key, val in params.items():
            if str(val).isdigit():  # handles numpy int64
                wandb_config[key] = {"value": int(val)}
            else:
                wandb_config[key] = {"value": val}
        return wandb_config

    def run(self):
        # overload scheduler run with special hyperopt logic

        def objective(params, worker_id):
            wandb.termlog(f"{LOG_PREFIX}Running objective with params: {params}")
            wandb_config = self._convert_search_space(params)
            run = self._create_run()
            srun = SweepRun(
                id=self._encode(run['id']),
                args=wandb_config,
                worker_id=worker_id,
            )
            wandb.termlog(f"{LOG_PREFIX}Hyperopt trials state:\n{self._trials}")
            self._add_to_launch_queue(srun)
            # wait for run to finish
            while True:
                try:
                    metrics = self._get_metrics_from_run(srun.id)
                    run_state = self._get_run_state(srun.id)
                    wandb.termlog(f"{LOG_PREFIX}{run_state=} {len(metrics)=}")

                    # TODO(gst): why does the state never change?
                    if not run_state.is_alive:
                        # done, log some metrics internally for visibility, then return
                        self._trials[srun.id] = {"min_loss": min(metrics), "num_metrics": len(metrics)}
                        del self._runs[srun.id]

                        return {
                            "loss": min(metrics),
                            "status": STATUS_OK,
                        }
                except Exception as e:
                    wandb.termwarn(f"Failed to poll run from public api: {str(e)}")
                    return {
                        "loss": -1,
                        "status": STATUS_FAIL,
                    }

                time.sleep(self._polling_sleep)

        # convert wandb params to hyperopt style
        search_space = self._make_search_space(self._sweep_config["parameters"])
        trials = Trials()
        objective_with_worker_id = partial(objective, worker_id=2)
        best_params = fmin(
            fn=objective_with_worker_id,
            space=search_space,
            algo=tpe.suggest,  # bayesian optimization
            max_evals=self._sweep_config.get("run_cap"),
            trials=trials,
            show_progressbar=False,
        )

        best_params = []
        wandb.termlog(f"{LOG_PREFIX}{best_params=} {trials.results=}")
        
        # Cleanup
        self.stop_sweep()
        self.exit()

    def _exit(self):
        pass

    def _get_next_sweep_run(self, worker_id: int) -> Optional[SweepRun]:
        pass

    def _load_state(self):
        pass

    def _poll(self):
        pass

    def _save_state(self):
        pass


if __name__ == "__main__":
    setup_scheduler(HyperoptScheduler)
