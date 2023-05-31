## Sweeps on launch -- custom scheduler jobs

This folder contains jobs that can be used as schedulers for launch sweeps. 

### Quickstart

To run an example using the wandb sweep logic, **ensure the target launch queue is active**, and run: 

```bash
wandb launch-sweep wandb_scheduler/example_configs/wandb_scheduler_config.yaml --queue <queue> --project <project>
```

### Custom scheduler jobs

The intended use is to clone the repo, modifying scripts with impunity, and then create jobs from them, with:

`python wandb_scheduler/wandb_scheduler.py` or `python optuna_scheduler/optuna_scheduler.py`

Note: There are three possible job-types that can be created from this script, in the following ways:
1. (default) Code-Artifact job. No flags required, logs the code in the current directory and constructs a job with the code.
2. Git job. Pass in `--enable_git` to create a git-based job, which uses the current git hash as the source of the job.
3. Container job. Build a container using the provided Dockerfile (`docker build . -t <image-name>`), and run `wandb launch -d <image-name>`. This creates a job that points to an image and will be pulled before executing (requires registry setup for remote launch resources).

Once a custom scheduler job is created, they can be used in launch-sweep configuration files in the following way:

```yaml
# template.yaml
method: custom
metric:
   name: validation_accuracy
   goal: maximize

# Use a training job here, NOT what we just created above
# This is the job to be used in the execution of the actual sweep
job: <TRAINING JOB>

# This is where we use the scheduler job that is created in the run above
scheduler:
   job: <SCHEDULER JOB>
   # Required if the job is sourced from an image, otherwise the scheduler
   # defaults to running in thread within the launch agent
   resource: local-container

   # When using a W&B backend (not Optuna), set the sweep method here
   settings:
      method: bayes


parameters:
   ...

```

3. Then, to run a launch-sweep, use the CLI command: 
   `wandb launch-sweep <config.yaml> --queue <queue> --project <project>`

### Important Notes: 

1. There are **two** different jobs that must be included in the sweep config! One is the training job, which can be created by running a local wandb run that has a call to `run.log_code()` (or is run inside of a container with the `WANDB_DOCKER` environment variable set). The second job is the one created by running the schedulers in this folder (job creation automatically handled). 
2. For the `wandb_scheduler.py`, set the `method` of the sweep (bayes, grid, random) in the `scheduler.settings.method` key. All sweep schedulers sourced from jobs require `method: custom` in the top-level of the sweep configuration.

### Optuna 

More information specific to the Wandb-Optuna sweep scheduler can be found in the `wandb/examples` repo [here](https://github.com/wandb/examples/launch/launch-sweeps/)
