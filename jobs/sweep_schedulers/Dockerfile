from python:3.10-bullseye

RUN pip install --upgrade pip
RUN pip install optuna
RUN pip install git+https://github.com/wandb/wandb.git@main#egg=wandb

COPY . .

# Fill me in with "wandb_scheduler.py" or "optuna_scheduler.py" etc.
ENTRYPOINT [ "python", "optuna_scheduler.py" ]
