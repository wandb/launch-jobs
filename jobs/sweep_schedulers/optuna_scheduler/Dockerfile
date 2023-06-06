# syntax=docker/dockerfile:1.4

# connectorx does not have linux ARM support
# https://github.com/sfu-db/connector-x/issues/240
FROM --platform=linux/amd64 python:3.9-slim
RUN apt update && apt install gcc -y

WORKDIR /launch

COPY --link requirements.txt ./
RUN pip install --upgrade pip 
RUN pip install -r requirements.txt

COPY optuna_wandb.py ./launch/optuna_wandb.py
COPY optuna_scheduler.py ./launch/optuna_scheduler.py

ENTRYPOINT [ "python", "./launch/optuna_scheduler.py" ]
