# syntax=docker/dockerfile:1.4

# connectorx does not have linux ARM support
# https://github.com/sfu-db/connector-x/issues/240
FROM --platform=linux/amd64 python:3.9-slim
RUN apt update && apt install gcc -y

WORKDIR /launch

COPY --link requirements.txt ./
RUN pip install -r requirements.txt

COPY optuna_wandb.py ./optuna_wandb.py

ENTRYPOINT [ "python", "optuna_scheduler.py" ]
