This folder contains jobs that can be used as schedulers for launch sweeps. 

The intended use is to clone the repo, modifying scripts with impunity, and then:

1. `python wandb_scheduler.py`
2. Use the job that is created in a sweep config file like: 

```yaml
method: bayes
metric:
   name: val_accuracy
   goal: maximize

# Use a training job here, NOT what we just created above
# This is the job to be used in the execution of the actual sweep
job: <TRAINING JOB>

# This is where we use the scheduler job that is created in the run above
scheduler:
   job: <SCHEDULER JOB>

parameters:
   ...

```

3. Then, to run a launch-sweep, use the CLI command: 
   `wandb launch-sweep <config.yaml> --queue <queue>

### Important Notes: 

1. There are **two** different jobs that must be included in the sweep config! One is the training job, which can be created by running a local wandb run that has a call to `run.log_code()`. The second job is the one created by running the schedulers in this folder. 
2. For the `wandb_scheduler`, use the `method` parameter in the sweep config as usual [bayes, grid, random]. For other schedulers, please set `method: custom`
