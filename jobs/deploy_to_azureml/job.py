import datetime
import logging
import os
import textwrap
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import wandb
import yaml
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
)
from azure.identity import DefaultAzureCredential
from pydantic import BaseModel
from rich.logging import RichHandler

logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

ArtifactType = Literal["onnx", "pytorch", "tensorflow"]


class Config(BaseModel):
    # Azure standard configs
    subscription_id: str
    resource_group: str
    workspace: str

    # The entity/project that each deployment will log to when the endpoint is hit
    entity: str
    project: str
    artifact_path: str  # Path to W&B artifact that contains model file
    keyvault_name: str  # Keyvault must contain the W&B API key under the key name "wandb-api-key"

    # Customizable endpoint configs
    endpoint_name: str = "endpoint-" + datetime.datetime.now().strftime("%m%d%H%M%f")
    endpoint_description: Optional[str] = None
    endpoint_tags: Optional[dict[str, Any]] = None

    # Customizable deployment configs
    deployment_name: str = "deployment-" + datetime.datetime.now().strftime(
        "%m%d%H%M%f"
    )
    deployment_instance_type: str = "Standard_DS3_v2"
    deployment_instance_count: int = 1
    image: str = "mcr.microsoft.com/azureml/inference-base-2204:latest"
    custom_image: bool = False  # Set to true if you want to bring your own container image with its own entrypoint

    # Specify both `artifact_type` and `artifact_model_name` to skip the model type inference step
    artifact_type: Optional[ArtifactType] = None
    artifact_model_name: Optional[str] = None

    # Do all the autogen and setup, but don't actually deploy the endpoint
    dry_run: bool = False


def infer_model_type_and_name(
    path: str,
) -> Tuple[Literal["pytorch", "tensorflow", "onnx", "unknown"], str]:
    # Define possible file extensions for each model type
    pytorch_extensions = {".pt", ".pth"}
    tensorflow_files = {"saved_model.pb"}
    onnx_extensions = {".onnx"}

    # Create a Path object for the directory
    dir_path = Path(path)

    # Check if the directory exists
    if not dir_path.exists():
        return "Directory does not exist."

    # Iterate over files in the directory using Path objects
    for file_path in dir_path.iterdir():
        if file_path.is_file():  # Ensure it's a file
            # Check the file extension against known model types
            if file_path.suffix in pytorch_extensions:
                return ("pytorch", file_path.name)
            elif file_path.name in tensorflow_files:
                return ("tensorflow", file_path.parent.name)
            elif file_path.suffix in onnx_extensions:
                return ("onnx", file_path.name)

    return ("unknown", "")


def generate_conda_yml(
    artifact_type: ArtifactType, conda_file: str = "conda.yml"
) -> None:
    """In future, this could pull from the conda env file of the run that generated the input model artifact."""

    deps_map = {
        "pytorch": ["torch", "timm"],
        "tensorflow": ["tensorflow"],
        "onnx": ["onnxruntime", "numpy"],
    }

    deps = deps_map.get(artifact_type, [])

    d = {
        "name": "deployment-env",
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.9",
            "pip",
            {
                "pip": [
                    "azureml-defaults",
                    "inference-schema[numpy-support]",
                    "azure-keyvault-secrets",
                    "azure-identity",
                    "azureml-inference-server-http",
                    "azure-ai-ml",
                    "azure-keyvault",
                    "wandb",
                    "numpy",
                    *deps,
                ]
            },
        ],
    }

    with open(conda_file, "w") as f:
        yaml.dump(d, f)


def generate_main_py(
    entity: str, project: str, model_name: str, artifact_type: ArtifactType
) -> None:
    # NOTE: Most of the examples use `score.py` as the scoring script, but I had to use `main.py`
    # because that's what the inference server defaults to and I wasn't sure how to change the file name.

    # Generate imports
    import_code = "# Generate imports"
    if artifact_type == "pytorch":
        import_code += """
        import torch
        """
    elif artifact_type == "tensorflow":
        import_code += """
        import tensorflow as tf
        """
    elif artifact_type == "onnx":
        import_code += """
        import numpy
        import onnxruntime
        """

    # Generate globals
    global_code = "# Generate globals"
    if artifact_type == "pytorch":
        global_code += """
            global model
            global wandb_run
            """
    elif artifact_type == "tensorflow":
        global_code += """
            global model
            global wandb_run
            """
    elif artifact_type == "onnx":
        global_code += """
            global session
            global wandb_run
            """

    # Generate model loading code
    loading_code = "# Generate model loading code"
    if artifact_type == "pytorch":
        loading_code += f"""
            model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "{model_name}")
            model = torch.load(model_path)
            """
    elif artifact_type == "tensorflow":
        loading_code += f"""
            # model_path = os.getenv("AZUREML_MODEL_DIR")
            model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "{model_name}")
            model = tf.saved_model.load(model_path)
            """
    elif artifact_type == "onnx":
        loading_code += f"""
            model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "{model_name}")
            session = onnxruntime.InferenceSession(model_path)
            """

    # Generate json to model input code
    converter_code = "# Generate json to model input code"
    if artifact_type == "pytorch":
        converter_code += """
                processed_data = torch.tensor(json_dict['data']).float()
                """
    elif artifact_type == "tensorflow":
        converter_code += """
                processed_data = tf.convert_to_tensor(json_dict['data'])
                """
    elif artifact_type == "onnx":
        converter_code += """
                processed_data = {}
                for k,v in json_dict.items():
                    processed_data[k] = numpy.array(v, dtype=numpy.float32)
                """

    # Generate inference code
    inference_code = "# Generate inference code"
    if artifact_type == "pytorch":
        inference_code += """
                output = model(processed_data).tolist()
                """
    elif artifact_type == "tensorflow":
        inference_code += """
                infer = model.signatures['serving_default']
                output = infer(processed_data)["predictions"].numpy().tolist()
                """
    elif artifact_type == "onnx":
        inference_code += """
                input_names = [i.name for i in session.get_inputs()]
                output_names = [i.name for i in session.get_outputs()]

                input_dict = {}
                for name in input_names:
                    assert name in processed_data
                    input_dict[name] = processed_data[name]

                output = session.run(output_names, input_dict)
                output = [v.tolist() for v in output]
        """

    text = textwrap.dedent(
        f"""
        import json
        import logging
        import os
        import wandb

        from azure.identity import ManagedIdentityCredential
        from azure.keyvault.secrets import SecretClient

        {import_code}

        def load_secrets():
            secret_clients = {{}}
            credential = ManagedIdentityCredential()

            for k, v in os.environ.items():
                if "KV_SECRET" in k:
                    trimmed_k = k.replace("KV_SECRET_", "")
                    secret_name, vault_url = v.split("@")

                    if vault_url in secret_clients:
                        secret_client = secret_clients[vault_url]
                    else:
                        secret_client = SecretClient(vault_url=vault_url, credential=credential)
                        secret_clients[vault_url] = secret_client

                    secret_value = secret_client.get_secret(secret_name).value
                    os.environ[trimmed_k] = secret_value


        def init():
            {global_code}

            load_secrets()

            {loading_code}

            wandb_run = wandb.init(
                {entity=},
                {project=},
                resume="allow",
                job_type="deploy-to-azureml",
            )
            logging.info("Init complete")


        def run(raw_data):
            logging.info("Request received")
            json_dict = json.loads(raw_data)
            output = None
            error = None

            table = wandb.Table(columns=["inputs", "outputs", "error"])
            try:
                {converter_code}
                {inference_code}
            except Exception as e:
                logging.info("Error processing request")
                error = str(e)
            else:
                logging.info("Successfully processed request")
            
            table.add_data(json_dict, output, error)

            wandb_run.log({{"inputs": json_dict, "outputs": output, "table": table}})
            return output

    """
    )

    with open("main.py", "w") as f:
        f.write(text)


# These will be entered in the Launch config
wandb_config = {
    "entity": "",
    "project": "",
    "subscription_id": "",
    "resource_group": "",
    "workspace": "",
    "keyvault_name": "",
    "artifact_path": "",
    "deployment_name": "",
    "dry_run": True,
}
run = wandb.init(config=wandb_config)

# if user provides custom configs via Launch, get them back
config = Config.model_validate(dict(run.config))
logging.info(f"Starting with {config=}")

if config.dry_run:
    logging.info("Starting in dry run mode")
    os.environ["WANDB_MODE"] = "disabled"

logging.info(f"Downloading {config.artifact_path=}")
api = wandb.Api()
art = api.artifact(config.artifact_path)
path = art.download()


if config.artifact_type is not None and config.artifact_model_name is not None:
    logging.info(
        f"Using user-provided {config.artifact_type=} and {config.artifact_model_name=}"
    )
    model_type = config.artifact_type
    model_name = config.artifact_model_name
else:
    logging.info("Inferring model type and name")
    model_type, model_name = infer_model_type_and_name(path)
    if model_type == "unknown":
        raise ValueError(
            "Model type could not be inferred, please specify it manually."
        )

logging.info(f"Using {model_type=} and {model_name=}")


logging.info("Configuring model and environment")
conda_file = "conda.yml"
generate_conda_yml(artifact_type=model_type, conda_file=conda_file)


logging.info("Generating a main.py")
generate_main_py(
    entity=config.entity,
    project=config.project,
    model_name=model_name,
    artifact_type=model_type,
)

ml_client = MLClient(
    DefaultAzureCredential(),
    config.subscription_id,
    config.resource_group,
    config.workspace,
)


logging.info("Creating endpoint...")
if (tags := config.endpoint_tags) is None:
    tags = {"source": "wandb-deploy"}

endpoint = ManagedOnlineEndpoint(
    name=config.endpoint_name,
    description=config.endpoint_description,
    auth_mode="key",
    tags=tags,
)
logging.info(f"{endpoint=}")
if not config.dry_run:
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()


logging.info("Creating deployment...")
dep_kwargs = {
    "name": config.deployment_name,
    "endpoint_name": config.endpoint_name,
    "instance_type": config.deployment_instance_type,
    "instance_count": config.deployment_instance_count,
}

if config.custom_image:
    dep_kwargs["environment"] = Environment(image=config.image)
else:
    dep_kwargs["environment"] = Environment(conda_file=conda_file, image=config.image)
    dep_kwargs["code_configuration"] = CodeConfiguration(
        code=".", scoring_script="main.py"
    )
    dep_kwargs["environment_variables"] = {
        "KV_SECRET_WANDB_API_KEY": f"wandb-api-key@https://{config.keyvault_name}.vault.azure.net"
    }

    if model_type == "tensorflow":
        dep_kwargs["model"] = Model(path=path)
    else:
        dep_kwargs["model"] = Model(path=os.path.join(path, model_name))


deployment = ManagedOnlineDeployment(**dep_kwargs)
logging.info(f"{deployment=}")
if not config.dry_run:
    ml_client.online_deployments.begin_create_or_update(deployment).result()

run.finish()
logging.info("All done")
