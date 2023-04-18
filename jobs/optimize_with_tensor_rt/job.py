import os
import time

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.python.compiler.tensorrt import trt_convert as trt

import wandb
from pathlib import Path

# Used to load example configs from wandb jobs repo.
# Is there a better way to handle this?
p = Path("config.yml")
if p.is_file():
    with open(p) as f:
        config = yaml.safe_load(f)


def benchmark(warmup_rounds, benchmarking_rounds, model, inp, metric_name):
    wandb.termlog("Warming up...")
    for i in range(warmup_rounds):
        model(inp)

    wandb.termlog("Benchmarking model...")
    for i in range(benchmarking_rounds):
        start = time.time()
        preds = model(inp)
        stop = time.time()
        time_ms = (stop - start) * 1000
        wandb.log({metric_name: time_ms, "benchmarking_step": i})


with wandb.init(config=config, job_type="optimize_model") as run:
    wandb.termlog("downloading model")
    model_dir = run.config.model.download()
    model = tf.keras.models.load_model(model_dir)

    wandb.termlog("converting model")
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=model_dir,
        conversion_params=trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=run.config.precision,
        ),
    )
    converter.convert()
    wandb.termlog("saving converted model")
    model_name = run.config.model.name.split("/")[-1].split(":")[0]
    optimized_model_name = f"{model_name}_trt_{run.config.precision}"
    converter.save(optimized_model_name)

    art2 = wandb.Artifact(
        optimized_model_name, f"model_optimized_{run.config.precision}"
    )
    art2.add_dir(optimized_model_name)
    run.log_artifact(art2)

    wandb.termlog("benchmarking models")
    if run.config.benchmark:
        inp = np.random.rand(*run.config.benchmark["input_shape"])
        inp = tf.convert_to_tensor(inp, dtype=tf.float32)

        benchmark(
            run.config.benchmark["warmup_rounds"],
            run.config.benchmark["benchmarking_rounds"],
            model,
            inp,
            "before_opt_inference_time_ms",
        )

        model2 = tf.keras.models.load_model(optimized_model_name)
        benchmark(
            run.config.benchmark["warmup_rounds"],
            run.config.benchmark["benchmarking_rounds"],
            model2,
            inp,
            "after_opt_inference_time_ms",
        )

    wandb.termlog("done")
