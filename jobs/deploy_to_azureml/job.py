import ast
import collections
import datetime
import logging
import os
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import onnx
import tensorflow as tf
import torch
import yaml
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
)
from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential
from pydantic import BaseModel, Field

import wandb


class ModelExtractor(ast.NodeVisitor):
    def __init__(self):
        self.imports = []
        self.classes = []

    def visit_Import(self, node):
        for alias in node.names:
            if alias.asname:
                self.imports.append(f"import {alias.name} as {alias.asname}")
            else:
                self.imports.append(f"import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            if alias.asname:
                self.imports.append(
                    f"from {node.module} import {alias.name} as {alias.asname}"
                )
            else:
                self.imports.append(f"from {node.module} import {alias.name}")
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        # This ensures the class definition is preserved as is, including method definitions.
        self.classes.append(ast.unparse(node))


def extract_model_code(file_path: str) -> str:
    with open(file_path, "r") as file:
        tree = ast.parse(file.read(), filename=file_path)

    extractor = ModelExtractor()
    extractor.visit(tree)

    return "\n".join(extractor.imports + extractor.classes)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["WANDB_ENABLE_RICH_LOGGING"] = "true"

if os.getenv("WANDB_ENABLE_RICH_LOGGING"):
    from rich.console import Console
    from rich.logging import RichHandler

    console = Console(width=120)
    logger.addHandler(
        RichHandler(rich_tracebacks=True, tracebacks_show_locals=True, console=console)
    )
else:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

ArtifactType = Literal["onnx", "pytorch", "tensorflow"]

time_now = datetime.datetime.now().strftime("%m%d%H%M%f")
default_image = "mcr.microsoft.com/azureml/inference-base-2204:latest"
pytorch_model_code_fname = "model_code.py"
scoring_fname = "score.py"
env_fname = "conda.yml"
request_file = "test.json"


class AzureConfigs(BaseModel):
    # Azure standard configs
    subscription_id: str
    resource_group: str
    workspace: str

    # Keyvault containing secrets like the W&B API Key
    keyvault: str


class WandbConfigs(BaseModel):
    # The entity/project that each deployment will log to when the endpoint is hit
    entity: str
    project: str


class DeployModelArtifact(BaseModel):
    path: str  # Path to W&B artifact that contains model file

    # Specify both `type` and `name` to skip the model type inference step
    type: Optional[ArtifactType] = None
    name: Optional[str] = None


class EndpointConfigs(BaseModel):
    name: str = f"endpoint-{time_now}"
    description: Optional[str] = None
    tags: dict[str, Any] = Field(default_factory=lambda: {"source": "wandb-deploy"})


class DeploymentConfigs(BaseModel):
    name: str = f"deployment-{time_now}"
    instance_type: str = "Standard_DS3_v2"
    instance_count: int = 1
    image: str = default_image
    # If not specified, generates score.py and requirements.txt from `scoring_artifact_path`
    scoring_artifact_path: Optional[str] = None


class Config(BaseModel):
    azure: AzureConfigs
    wandb: WandbConfigs
    deploy_model_artifact: DeployModelArtifact

    endpoint: EndpointConfigs = Field(default_factory=EndpointConfigs)
    deployment: DeploymentConfigs = Field(default_factory=DeploymentConfigs)

    # Do all the autogen and setup, but don't actually deploy the endpoint
    dry_run: bool = False

    @property
    def deployed_model_path(self):
        workspace = self.azure.workspace
        endpoint = self.endpoint.name
        deployment = self.deployment.name

        return os.path.join(workspace, endpoint, deployment)


class InferredModel(BaseModel):
    type: ArtifactType
    name: str


def infer_model(path: str) -> InferredModel:
    # Define possible file extensions for each model type
    pytorch_extensions = {".pt", ".pth"}
    tensorflow_files = {"saved_model.pb"}
    onnx_extensions = {".onnx"}

    # Create a Path object for the directory
    dir_path = Path(path)

    # Check if the directory exists
    if not dir_path.exists():
        raise Exception("Directory does not exist")

    def walk(path: Path) -> Optional[InferredModel]:
        if path.is_file():
            if path.suffix in pytorch_extensions:
                return InferredModel(type="pytorch", name=path.name)
            elif path.suffix in onnx_extensions:
                return InferredModel(type="onnx", name=path.name)
            elif path.name in tensorflow_files:
                # Get the SavedModel parent folder
                if path.parent.name == "saved_model":
                    return InferredModel(type="tensorflow", name=path.parent.name)
                elif path.parent.parent.name == "saved_model":
                    return InferredModel(
                        type="tensorflow",
                        name=f"{path.parent.parent.name}/{path.parent.name}",
                    )
        elif path.is_dir():
            for p in path.iterdir():
                if (result := walk(p)) is not None:
                    return result
        return None

    if (result := walk(dir_path)) is None:
        raise Exception("Unable to infer model type, please specify manually")

    return result

    # # Iterate over files in the directory using Path objects
    # for file_path in dir_path.iterdir():
    #     if file_path.is_file():  # Ensure it's a file
    #         # Check the file extension against known model types
    #         if file_path.suffix in pytorch_extensions:
    #             return InferredModel(type="pytorch", name=file_path.name)
    #         # elif file_path.name in tensorflow_files:
    #         #     return InferredModel(type="tensorflow", name=file_path.parent.name)
    #         elif file_path.suffix in onnx_extensions:
    #             return InferredModel(type="onnx", name=file_path.name)
    #     elif file_path.is_dir():
    #         if file_path.name in tensorflow_files:
    #             return InferredModel(type="tensorflow", name=file_path.name)

    raise Exception("Unable to infer model type, please specify manually")


def generate_conda_yml(
    artifact_type: ArtifactType,
    fname: str = "conda.yml",
    *,
    config_artifact: Optional[str] = None,
) -> None:
    """In future, this could pull from the conda env file of the run that generated
    the input model artifact.
    """
    fall_back_to_defaults = True

    # Try to get requirements from provided artifact
    if config_artifact is not None:
        api = wandb.Api()
        art = api.artifact(config_artifact)
        run = art.logged_by

        for f in run.files():
            if f.name == "requirements.txt":
                path = f.download()
                with open(path.name) as ff:
                    deps = ff.read().splitlines()

                fall_back_to_defaults = False
                break

    # Otherwise, fall back to defaults
    if fall_back_to_defaults:
        logger.info("`requirements.txt` not found, falling back to defaults for env")
        base_deps = ["onnx", "onnxruntime", "numpy"]
        deps_map = {
            "pytorch": ["torch", "timm", "transformers", "accelerate", *base_deps],
            "tensorflow": ["tensorflow", "transformers", "accelerate", *base_deps],
            "onnx": base_deps,
        }
        deps = deps_map[artifact_type]

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

    with open(fname, "w") as f:
        yaml.dump(d, f)


def infer_pytorch_model_code(model_art: wandb.Artifact) -> str:
    if (run := model_art.logged_by()) is None:
        raise Exception("Model artifact does not have a generating run")

    if (code_path := run.metadata.get("codePath")) is None:
        raise Exception("Run is missing codePath; can't infer imports")

    for f in run.files():
        if f.name == f"code/{code_path}":
            break
    else:
        raise Exception("Code file not found")

    code_path = f.download()
    model_code = extract_model_code(code_path.name)

    with open(pytorch_model_code_fname, "w") as f:
        f.write(model_code)

    return model_code


def generate_scoring_file(
    entity: str,
    project: str,
    model_name: str,
    artifact_type: ArtifactType,
    fname: str,
    model_art: Optional[wandb.Artifact] = None,
) -> None:
    # Try to get scoring file from provided artifact
    # Otherwise fall back to defaults

    # Generate imports
    if artifact_type == "pytorch":
        import_code = textwrap.dedent(
            """
            import torch
            """
        )
    elif artifact_type == "tensorflow":
        import_code = """
            import tensorflow as tf
            """
    elif artifact_type == "onnx":
        import_code = """
            import numpy as np
            import onnx
            import onnxruntime
            """
    import_code = textwrap.dedent(import_code)

    # Generate model loading code
    if artifact_type == "pytorch":
        model_code = infer_pytorch_model_code(model_art)
        loading_code = "\n" + textwrap.dedent(model_code)
        loading_code += textwrap.dedent(
            f"""
            model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "{model_name}")

            global model
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
            model.eval()
            # model = torch.load(model_path)
            """
        )
    elif artifact_type == "tensorflow":
        loading_code = f"""
            model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "{model_art.name}", "{model_name}")

            global model
            model = tf.saved_model.load(model_path)
            """
    elif artifact_type == "onnx":
        loading_code = f"""
            model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "{model_name}")
            """
        loading_code += r"""
            global session
            session = onnxruntime.InferenceSession(model_path)
            
            global model
            model = onnx.load(model_path)
            
            global dtype_map
            dtype_map = {
                onnx.TensorProto.FLOAT: np.float32,
                onnx.TensorProto.DOUBLE: np.float64,
                onnx.TensorProto.INT32: np.int32,
                onnx.TensorProto.INT64: np.int64,
            }
            """
    loading_code = textwrap.dedent(loading_code)

    # Generate inference code
    if artifact_type == "pytorch":
        inference_code = r"""
            model_inputs = {k: torch.tensor(v) for k, v in json_dict.items()}
            output = model(**model_inputs).tolist()
            """
    elif artifact_type == "tensorflow":
        inference_code = r"""
            model_inputs = {k: tf.convert_to_tensor(v) for k, v in json_dict.items()}
            infer = model.signatures['serving_default']
            output = infer(**model_inputs)["predictions"].numpy().tolist()
            """
    elif artifact_type == "onnx":
        inference_code = r"""
            model_inputs = {}
            for input_tensor in model.graph.input:
                input_name = input_tensor.name
                input_dtype = input_tensor.type.tensor_type.elem_type
                if (expected_dtype := dtype_map.get(input_dtype)) is None:
                    raise ValueError(f"Found unsupported {input_dtype=}")

                # Convert the data to the appropriate numpy data type
                if input_name in json_dict:
                    converted_data = np.asarray(json_dict[input_name], dtype=expected_dtype)
                    model_inputs[input_name] = converted_data
                else:
                    raise ValueError(f"Input data for '{input_name}' not provided in JSON.")

            output = [x.tolist() for x in session.run(None, model_inputs)]
            """
    inference_code = textwrap.dedent(inference_code)

    text = textwrap.dedent(
        f"""
import json
import logging
import os
import wandb

from azure.identity import ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient

# GENERATED IMPORTS
# -----------------
{import_code}
# -----------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

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
    global wandb_run
    wandb_run = None
    
    try:
        load_secrets()
    except Exception as e:
        logger.warning("Error when loading secrets.  Logging to W&B disabled")
    else:
        wandb_run = wandb.init(
            {entity=},
            {project=},
            resume="allow",
            job_type="deploy-to-azureml",
        )

    # GENERATED MODEL LOADING CODE
    # ----------------------------
    {textwrap.indent(loading_code, " "*4)}
    # ----------------------------

    logger.info("Init complete")


def run(raw_data):
    logger.info("Request received")
    json_dict = json.loads(raw_data)
    output = None
    error = None
    
    try:
        # GENERATED INFERENCE CODE
        # ------------------------
        {textwrap.indent(inference_code, " "*8)}
        # ------------------------
    except Exception as e:
        logger.info("Error processing request", e)
        error = str(e)
    else:
        logger.info("Successfully processed request")

    if wandb_run is not None:
        table = wandb.Table(columns=["inputs", "outputs", "error"])
        table.add_data(json_dict, output, error)
        wandb_run.log({{"table": table}})

    return output
    """
    )

    with open(fname, "w") as f:
        f.write(text)

    subprocess.run(["ruff", "check", fname, "--select", "I", "--fix"])
    subprocess.run(["ruff", "format", fname])


run = wandb.init()
config = Config.model_validate(dict(run.config))

logger.info(f"Starting with {config=}")
if config.dry_run:
    logger.info("Starting in dry run mode")
    os.environ["WANDB_MODE"] = "disabled"

logger.info(f"Downloading {config.deploy_model_artifact.path=}")
api = wandb.Api()
model_art = api.artifact(config.deploy_model_artifact.path)
path = model_art.download(skip_cache=True)
run.use_artifact(model_art)


# Get model type and name
if (
    config.deploy_model_artifact.type is not None
    and config.deploy_model_artifact.name is not None
):
    model_type = config.model_atifact.type
    model_name = config.deploy_model_artifact.name
else:
    logger.info("Inferring model type and name...")
    inferred_model = infer_model(path)
    model_type = inferred_model.type
    model_name = inferred_model.name

logger.info(f"Using {model_type=} and {model_name=}")


# Check model compat
if model_type == "pytorch":
    pass
    # torch models need the class, so either user provides or we infer

    model_path = os.path.join(path, model_name)
    try:
        model = torch.load(model_path)

    # Loaded a pickle: Will error because the class is not defined in this file, but
    # it means the user has the correct type of pytorch file -- continue on.
    except AttributeError:
        pass
    else:
        # We can't support state dicts because we don't know what the model class is
        if isinstance(model, collections.OrderedDict):
            raise ValueError(
                "Model appears to be a state dict.  Please save model artifact as a pickle (torch.save without state_dict()) and try again."
            )


# Get score.py and environment

if config.deployment.scoring_artifact_path is not None:
    scoring_art = api.artifact(config.deployment.scoring_artifact_path)
    path2 = Path(scoring_art.download(skip_cache=True))
    shutil.move(path2 / scoring_fname, scoring_fname)
    shutil.move(path2 / env_fname, env_fname)
    run.use_artifact(scoring_art)
else:
    logger.info("Generating score.py and environment...")
    logger.info("Generating score.py...")
    generate_scoring_file(
        entity=config.wandb.entity,
        project=config.wandb.project,
        model_name=model_name,
        artifact_type=model_type,
        fname=scoring_fname,
        model_art=model_art,
    )
    logger.info(f"Generating {env_fname}...")
    generate_conda_yml(artifact_type=model_type, fname=env_fname)

    scoring_art = wandb.Artifact("score_env", "score_env")
    scoring_art.add_file(scoring_fname)
    scoring_art.add_file(env_fname)
    if model_type == "pytorch":
        scoring_art.add_file(pytorch_model_code_fname)

    run.use_artifact(scoring_art)


# Create the endpoint
logger.info("Starting create endpoint process (this may take a while...)")
ml_client = MLClient(
    DefaultAzureCredential(),
    config.azure.subscription_id,
    config.azure.resource_group,
    config.azure.workspace,
)

endpoint = ManagedOnlineEndpoint(
    name=config.endpoint.name,
    description=config.endpoint.description,
    auth_mode="key",
    tags=config.endpoint.tags,
)
logger.info(f"Creating {endpoint=}")
if not config.dry_run:
    try:
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    except HttpResponseError as e:
        logger.error("Error creating endpoint: " + str(e))


# Create the deployment
logger.info("Starting create deployment process (this may take a while...)")
d = {
    "name": config.deployment.name,
    "endpoint_name": config.endpoint.name,
    "instance_type": config.deployment.instance_type,
    "instance_count": config.deployment.instance_count,
}

if config.deployment.image != default_image:
    d["environment"] = Environment(image=config.deployment.image)
else:
    d["environment"] = Environment(conda_file=env_fname, image=config.deployment.image)
    d["code_configuration"] = CodeConfiguration(code=".", scoring_script="score.py")
    d["environment_variables"] = {
        "KV_SECRET_WANDB_API_KEY": f"wandb-api-key@https://{config.azure.keyvault}.vault.azure.net"
    }

    if model_type == "tensorflow":
        d["model"] = Model(path=path)
    else:
        d["model"] = Model(path=os.path.join(path, model_name))

deployment = ManagedOnlineDeployment(**d)
logger.info(f"Creating {deployment=}")
if not config.dry_run:
    try:
        ml_client.online_deployments.begin_create_or_update(deployment).result()
    except HttpResponseError as e:
        logger.error("Error creating deployment: " + str(e))

logger.info("Checking model inputs...")
if model_type == "pytorch":
    logger.info("Unable to infer pytorch model inputs (check model manually)")

elif model_type == "tensorflow":
    model_path = os.path.join(path, model_name)
    model = tf.saved_model.load(model_path)

    sig = model.signatures["serving_default"]
    input_details = sig.structured_input_signature[1]

    input_shapes = {name: spec.shape for name, spec in input_details.items()}
    logger.info(
        f"Your model ({config.deployed_model_path}) is expecting a json payload with {input_shapes=}"
    )

elif model_type == "onnx":
    dtype_map = {
        onnx.TensorProto.FLOAT: np.float32,
        onnx.TensorProto.DOUBLE: np.float64,
        onnx.TensorProto.INT32: np.int32,
        onnx.TensorProto.INT64: np.int64,
    }

    model_path = os.path.join(path, model_name)
    model = onnx.load(model_path)

    input_shapes = {}
    for input_tensor in model.graph.input:
        name = input_tensor.name
        shape = [d.dim_value for d in input_tensor.type.tensor_type.shape.dim]
        input_shapes[name] = shape
    logger.info(
        f"Your model ({config.deployed_model_path}) is expecting a json payload with {input_shapes=}"
    )
    logger.warning("ONNX models require the inputs to be in order!")

logger.info("Finished deployment!")
run.finish()
