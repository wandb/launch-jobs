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

### Important Notes: 

1. For the `wandb_scheduler`, use the `method` parameter in the sweep config as usual [bayes, grid, random]. For other schedulers, please set `method: custom`
