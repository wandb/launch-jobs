## Sweeps on launch -- custom scheduler jobs

This folder contains jobs that can be used as schedulers for launch sweeps. For examples using jobs that have already been created, head over to the `wandb/examples` repo [here](https://github.com/wandb/examples/tree/master/examples/launch/launch-sweeps).

NOTE: These jobs won't appear on the jobs page because they behave a bit differently than vanilla jobs.

### Quickstart using premade jobs

To run an example using the Wandb sweep logic, **ensure the target launch queue is active** (more info [here](https://docs.wandb.ai/guides/launch/quickstart)), and run:

```bash
wandb launch-sweep wandb_scheduler/example_configs/basic.yaml --queue <queue> --project <project> --entity <entity>
```

This basic config contains a few example parameters to get a scheduler running, shown below:

```yaml
# basic.yaml
scheduler:
  job: "wandb/jobs/Wandb Scheduler Image Job:latest"
  resource: local-container # required for image jobs

  settings:
    method: bayes # required to specify method here for Wandb scheduler

job: "wandb/jobs/Fashion MNIST Train Job:latest"
run_cap: 5
metric:
  goal: minimize
  name: loss

parameters:
  epochs:
    values: [10, 25, 100]
  learning_rate:
    min: 0.00001
    max: 1.0
```

Or, to run the pre-made Optuna Scheduler job, run:

```bash
wandb launch-sweep optuna_scheduler/example_configs/basic.yaml --queue <queue> --project <project> --entity <entity>
```

More information specific to the Optuna sweep scheduler can be found in the `wandb/examples` repo [here](https://github.com/wandb/examples/tree/master/examples/launch/launch-sweeps/optuna-scheduler)

### Custom scheduler jobs

While the above example uses a pre-made scheduler job (`'wandb/jobs/Wandb Scheduler Image Job:latest'`), it is also possible to create completely custom sweep scheduler jobs. Using a few simple utility methods, available in the `Scheduler` class within the `wandb` python package, any algorithm for sweep hyperparameter suggestion can be turned into a controller for sweeps. Easily customize the Wandb and Optuna Schedulers using this repo. Clone the repo, modify the `*_scheduler` scripts with impunity, and then create jobs from them, with:

`python wandb_scheduler/wandb_scheduler.py` or `python optuna_scheduler/optuna_scheduler.py`

For a specific example of a simple modification to the `wandb_scheduler.py` file, check out the example in `wandb/examples` [here](https://github.com/wandb/examples/tree/master/examples/launch/launch-sweeps/custom-scheduler).

Note: There are three different types of jobs (code, git, image):

1. (default) Code-Artifact job. No flags required, logs the code in the current directory and constructs a job with the code.
2. Git job. Pass in `--enable_git` to create a git-based job, which uses the current git hash as the source of the job.
3. Image job. Build a container using the provided Dockerfile (`docker build . -t <image-name>`), and run `wandb launch -d <image-name>`. This creates a job that points to an image and will be pulled before executing (requires registry setup for remote launch resources).

Once a custom scheduler job is created, they can be used in launch-sweep configuration files in the following way:

```yaml
metric:
  name: validation_accuracy
  goal: maximize

# Use a training job here, NOT what we just created above
# This is the job to be used in the execution of the actual sweep
job: <TRAINING JOB>

scheduler:
  # Use the scheduler job that is created by running python *_scheduler.py
  job: <SCHEDULER JOB>
  # Required if the job is sourced from an image, otherwise the scheduler
  # defaults to running in thread within the launch agent
  resource: local-container

  # Set settings for the scheduler here
  # this is passed into scheduler wandb run, and can be accessed in the scheduler with
  # self._wandb_run.config.get("settings", {}) (or in the Optuna scheduler self._optuna_config)
  settings:
    # When using a W&B backend (WandbScheduler from wandb_scheduler.py)
    #    set the sweep method here
    method: bayes

parameters: ...
```

3. Then, to run a launch-sweep, use the CLI command:
   `wandb launch-sweep <config.yaml> --queue <queue> --project <project> --entity <entity>`

### Important Notes:

1. There are **two** different jobs that must be included in the sweep config! One is the training job, which can be created by running a local wandb run that has a call to `run.log_code()` (or is run inside of a container with the `WANDB_DOCKER` environment variable set). The second job is the one created by running the schedulers in this folder (job creation automatically handled).
2. For the `wandb_scheduler.py`, set the `method` of the sweep (bayes, grid, random) in the `scheduler.settings.method` key. All sweep schedulers sourced from jobs require `method: custom` in the top-level of the sweep configuration.
