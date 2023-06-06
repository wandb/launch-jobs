"""Example training job"""
import wandb

import time
import random
import argparse
from typing import Optional, Any


def train(project: Optional[str], entity: Optional[str], **kwargs: Any):
    run = wandb.init(project=project, entity=entity)

    # get config, could be set from sweep scheduler
    train_config = run.config

    # TODO(gst): can we get metric from the run?
    metric = train_config.get("metric", "loss")

    # get training parameters
    epochs = train_config.get("epochs", 100)
    sleep = train_config.get("sleep", 0.1)
    learning_rate = train_config.get("learning_rate", 0.1)
    bias = train_config.get("bias", 0.0)

    # iterate through epochs, mocking a model fitting process
    for i in range(epochs):
        metric_val = _fit_model(learning_rate, i+1, bias)

        # log to wandb, print to console, and sleep to mimic compute
        run.log({metric: metric_val})
        print(f"Epoch: {i}, {metric}: {metric_val}")
        time.sleep(sleep)


def _fit_model(learning_rate: float, i: int, bias: float = 0.0):
    return max((1000/(i**1.5) + (5 * learning_rate + random.random())) + bias, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", "-e", type=str, default=None)
    parser.add_argument("--project", "-p", type=str, default=None)
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()