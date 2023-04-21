import torch
import os


def model_fn(model_dir, context):
    model = Model()  # noqa: F821
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model
