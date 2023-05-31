from python:3.10-bullseye

RUN pip install --upgrade pip
RUN pip install optuna
RUN pip install git+https://github.com/wandb/wandb.git@main#egg=wandb

COPY . .

ENTRYPOINT [ "python", "wandb_scheduler.py" ]
