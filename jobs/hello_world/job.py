import wandb

settings = wandb.Settings(disable_git=True)

with wandb.init(settings=settings) as run:
    run.log({"hello": "world"})
