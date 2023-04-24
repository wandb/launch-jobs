"""
Download a wandb artifact and deploy it to Sagemaker Endpoints.
"""

import shutil
import tarfile

import click
from sagemaker import Session

import wandb


def wandb_termlog_heading(text):
    return wandb.termlog(click.style("sagemaker job: ", fg="green") + text)


def err_raise(msg, e=ValueError):
    wandb.termerror(msg)
    raise e(msg)


required_kwargs = {
    "pytorch": {
        "artifact",
        "framework_version",
        "python_version",
        "sagemaker_role",
        "sagemaker_bucket",
        "instance_type",
    },
    "tensorflow": {
        "artifact",
        "framework_version",
        "sagemaker_role",
        "sagemaker_bucket",
        "instance_type",
    },
}

supported_frameworks = {"tensorflow", "pytorch"}


settings = wandb.Settings(disable_git=True)

with wandb.init(settings=settings) as run:
    if run.config["framework"] not in supported_frameworks:
        err_raise(
            f"Model type: {run.config['framework']} not supported.  Model type must be one of {run.config['supported_frameworks']}"
        )
    if run.config["framework"] == "tensorflow":
        from sagemaker.tensorflow.model import TensorFlowModel
    if run.config["framework"] == "pytorch":
        from sagemaker.pytorch.model import PyTorchModel

    unset_kwargs = {
        k for k in required_kwargs[run.config["framework"]] if not run.config.get(k)
    }
    if unset_kwargs:
        err_raise(
            f"Config keys required but not set: {unset_kwargs}.  Please specify in the run config dict!"
        )

    wandb_termlog_heading("Downloading artifact from wandb")
    path = run.config["artifact"].download()

    wandb_termlog_heading("Creating temp directory for sagemaker model")
    name_ver = path.split("/")[-1]
    name, ver = name_ver.split(":v")
    target = f"temp/{name}/{name}/{ver}"
    shutil.copytree(path, target)
    if run.config["framework"] == "pytorch":
        shutil.copyfile("./inference.py", f"temp/{name}/{name}/{ver}/inference.py")

    model_str = f"{name}-{ver}"
    model_tar = f"temp/{model_str}.tar.gz"
    with tarfile.open(model_tar, mode="w:gz") as archive:
        archive.add(target, recursive=True)

    wandb_termlog_heading("Uploading model to S3")
    session = Session()
    model_data = session.upload_data(
        bucket=run.config["sagemaker_bucket"], path=model_tar, key_prefix=model_str
    )

    wandb_termlog_heading(
        "Deploy model to Sagemaker Endpoints (this may take a while...)"
    )
    if run.config["framework"] == "tensorflow":
        sm_model = TensorFlowModel(
            model_data=model_data,
            framework_version=run.config["framework_version"],
            role=run.config["sagemaker_role"],
            **run.config["sagemaker_model_setup_kwargs"],
        )

    if run.config["framework"] == "pytorch":
        sm_model = PyTorchModel(
            entry_point="inference.py",
            py_version=run.config["python_version"],
            model_data=model_data,
            framework_version=run.config["framework_version"],
            role=run.config["sagemaker_role"],
            **run.config["sagemaker_model_setup_kwargs"],
        )

    predictor = sm_model.deploy(
        initial_instance_count=run.config["instance_count"],
        instance_type=run.config["instance_type"],
        # tags=run.config["sagemaker_deploy_tags"],
        **run.config["sagemaker_model_deployment_kwargs"],
    )

    wandb_termlog_heading(f"Successfully deployed endpoint: {predictor.endpoint}")
    run.log({"sagemaker_endpoint": predictor.endpoint})
