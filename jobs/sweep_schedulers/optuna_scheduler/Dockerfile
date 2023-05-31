from python:3.10-bullseye

RUN pip install --upgrade pip
RUN pip install optuna scipy
RUN pip install git+https://github.com/wandb/wandb.git@main#egg=wandb

COPY . .

ENTRYPOINT [ "python", "optuna_scheduler.py" ]
