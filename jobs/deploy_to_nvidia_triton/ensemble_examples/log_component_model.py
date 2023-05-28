import wandb
import random
import math

config = {
    "learning_rate": 0.01 * random.random(),
    "batch_size": 128,
    "momentum": 0.1 * random.random(),
    "dropout": 0.4 * random.random(),
    "dataset": ["hello", "world", 2],
    "model": "trained-model",
}


def log_component_model(component):
    with wandb.init(
        project="red-bull-triton", config=config, job_type=f"generate_{component}"
    ) as run:
        EVAL_STEPS = 1000
        # Log metrics and checkpoints at N steps
        displacement1 = random.random() * 2
        displacement2 = random.random() * 4
        for step in range(EVAL_STEPS):
            run.log(
                {
                    "acc": 0.1
                    + 0.4
                    * (
                        math.log(1 + step + random.random())
                        + random.random() * run.config.learning_rate
                        + random.random()
                        + displacement1
                        + random.random() * run.config.momentum
                    ),
                    "val_acc": 0.1
                    + 0.5
                    * (
                        math.log(1 + step + random.random())
                        + random.random() * run.config.learning_rate
                        - random.random()
                        + displacement1
                    ),
                    "loss": 0.1
                    + 0.08
                    * (
                        3.5
                        - math.log(1 + step + random.random())
                        + random.random() * run.config.momentum
                        + random.random()
                        + displacement2
                    ),
                    "val_loss": 0.1
                    + 0.04
                    * (
                        4.5
                        - math.log(1 + step + random.random())
                        + random.random() * run.config.learning_rate
                        - random.random()
                        + displacement2
                    ),
                }
            )
        art = wandb.Artifact(component, type="model")
        art.add_dir(f"ensemble_model/{component}")
        run.log_artifact(art)
        run.link_artifact(art, "model-registry/Text Detection")


def update_file(fpath):
    with open(fpath) as f:
        text = f.readlines()
        text = text + ["\n"]

    with open(fpath, "w") as f:
        f.writelines(text)


def component2fpath(component):
    return f"ensemble_model/{component}/config.pbtxt"


component = "text_detection"

fpath = component2fpath(component)
update_file(fpath)
log_component_model(component)
