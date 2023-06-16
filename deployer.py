"""
Utility for deploying W&B Launch docker jobs to a job repo
"""

import platform
import re
from pathlib import Path

import boto3
import click
import torch
import yaml

import wandb
from wandb.sdk.internal.internal_api import Api as InternalApi
from wandb.sdk.launch import launch, launch_add

api = wandb.Api()
iapi = InternalApi()
is_m1 = platform.machine() == "arm64" and platform.system() == "Darwin"
has_gpu = torch.cuda.is_available()

deploy_method_options = ["launch-agent", "launch-run"]


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--image", prompt="Enter image name, including tag (e.g. wandb/example:80ff320)"
)
@click.option("--config_path", prompt="Enter path to run config yml file")
@click.option(
    "--job_artifact_name", prompt="Enter a custom name for this job", default=""
)
@click.option(
    "--job_artifact_description",
    prompt="Enter a custom description for this job",
    default="",
)
@click.option(
    "--entity", "-e", prompt="Enter run's target entity", default="launch-test"
)
@click.option("--project", "-p", prompt="Enter run's target project", default="jobs")
@click.option(
    "--deploy_method", type=click.Choice(deploy_method_options), default="launch-run"
)
@click.option("--queue_name", default=None)
@click.option("--resource_args_path", default=None)
def deploy(
    image,
    config_path,
    job_artifact_name,
    job_artifact_description,
    entity,
    project,
    deploy_method,
    queue_name,
    resource_args_path,
):
    wandb.termlog(f"Deploying {image} with {config_path}...")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if deploy_method == "launch-agent":
        launch_add.launch_add(
            docker_image=image,
            name=config.get("run_name"),
            config={"overrides": {"run_config": config.get("config", {})}},
            queue_name=queue_name,
            entity=entity,
            project=project,
        )

    elif deploy_method == "launch-run":
        resource_args = None
        if resource_args_path is not None:
            with open(resource_args_path) as f:
                resource_args = yaml.safe_load(f)

        launch.run(
            iapi,
            docker_image=image,
            name=config["run_name"],
            config={"overrides": {"run_config": config["config"]}},
            resource="local-container",
            resource_args=resource_args,
            entity=entity,
            project=project,
        )

    # TODO: Rename artifact to something user friendly
    if job_artifact_name or job_artifact_description:
        ...


@cli.command()
@click.option("--registry_path", default="registry.yaml")
@click.option(
    "--tag",
    "-t",
    prompt="Enter tag for images to be deployed (git sha preferred)",
    default="main",
)
@click.option(
    "--entity", "-e", prompt="Enter run's target entity", default="launch-test"
)
@click.option("--project", "-p", prompt="Enter run's target project", default="jobs")
@click.option(
    "--deploy_method", type=click.Choice(deploy_method_options), default="launch-run"
)
@click.option(
    "--queue_name",
    prompt="Enter queue name (only if you selected `launch-agent`)",
    default="",
)
@click.option(
    "--resource_args_path",
    prompt="Enter agent resource args yaml path (only if you selected `launch-run`)",
    default="resource_args.yaml",
)
def deploy_everything(
    registry_path,
    tag,
    entity,
    project,
    deploy_method,
    queue_name,
    resource_args_path,
):
    if not click.confirm(
        "You are about to deploy many jobs...  This may take a while!  Please confirm:"
    ):
        return

    registry = get_registry(registry_path)
    for job_dir, metadata in registry.items():
        *dir_parts, job_name = job_dir.split("/")
        image = f"wandb/job_{job_name}:{tag}"

        if is_m1 and job_name == "sql_query":
            wandb.termwarn(f"Skipping {job_name} on M1: Upstream connectorx issue")
            continue

        if not has_gpu and job_name.startswith("gpu_"):
            wandb.termwarn(f"Skipping {job_name}: No GPU detected.")
            continue

        configs_path = Path("jobs", *dir_parts, job_name, "configs")
        for config_path in configs_path.glob("*.yml"):
            ctx = click.Context(deploy)
            ctx.invoke(
                deploy,
                image=image,
                config_path=config_path,
                entity=entity,
                project=project,
                deploy_method=deploy_method,
                queue_name=queue_name,
                resource_args_path=resource_args_path,
            )

        # TODO: Rename artifact to something user friendly using job_artifact_name and job_artifact_description


@cli.command()
def cleanup_sagemaker():
    wandb.termlog("Cleaning up sagemaker endpoints...")

    sm = boto3.client("sagemaker")
    response = sm.list_endpoints()
    endpoints = response["Endpoints"]

    for endpoint in endpoints:
        try:
            sm.delete_endpoint(EndpointName=endpoint["EndpointName"])
        except Exception as e:
            wandb.termerror(f"Problem deleting {endpoint['EndpointName']} {e}")
        else:
            wandb.termlog(f"Successfully deleted {endpoint['EndpointName']}")


def image_name_to_artifact_name(s, alias="latest"):
    result = re.sub(r"[:/]", "_", s)
    result = f"job-{result}:{alias}"
    return result


def _traverse_dict(d, path=None):
    if path is None:
        path = []
    output = {}
    for k, v in d.items():
        new_path = path + [k]
        if isinstance(v, dict):
            if "name" in v and "desc" in v:
                output["/".join(new_path)] = {"name": v["name"], "desc": v["desc"]}
            else:
                output.update(_traverse_dict(v, new_path))
    return output


def get_registry(fname: str = "registry.yaml"):
    with open(fname) as f:
        registry = yaml.safe_load(f)
        return _traverse_dict(registry)


if __name__ == "__main__":
    cli()
