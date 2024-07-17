import importlib.util
import site
import tempfile
import wandb

from huggingface_hub import snapshot_download
from omegaconf import DictConfig


def get_script_path(recipe_name: str):
    """Get recipe script path from recipe name.

    Torchtune installs recipe files including configs and scripts to a `recipes`
    directory in the site-packages directory.
    """
    site_packages = site.getsitepackages()[0]
    return site_packages + f"/recipes/{recipe_name}.py"


def load_recipe(recipe_name: str, recipe_class: str):
    """Load recipe module given the recipe name and class class.

    This function loads the recipe module from the recipe script path and returns
    the recipe. Direct import is not possible due to an exception raised in
    `recipes/__init__.py`.
    """
    script_path = get_script_path(recipe_name)
    spec = importlib.util.spec_from_file_location(recipe_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__dict__[recipe_class]


def execute(
    model: str,
    recipe_name: str,
    recipe_class: str,
    config: DictConfig,
    dataset_artifact: str,
):
    """Run a given recipe, model, and config.
    
    This function downloads the model snapshot, sets the model directory in the
    config, creates a temporary output directory, and runs the recipe. The recipe
    is loaded from the recipe script path and the recipe class is instantiated with
    the config. 

    Args:
        model (str): Hugging Face model hub name.
        recipe_name (str): Recipe name.
        recipe_class (str): Recipe class name.
        config (DictConfig): Recipe configuration.
        dataset_artifact (str): Dataset artifact name.
    """
    with wandb.init(config={"dataset_artifact": dataset_artifact}) as run:
        dataset = run.use_artifact(run.config.dataset_artifact)
        config.data_dir = dataset.download()
        model_dir = snapshot_download(model)
        config["model_dir"] = model_dir
        with tempfile.TemporaryDirectory() as outdir:
            config["output_dir"] = outdir
            recipe_constructor = load_recipe(recipe_name, recipe_class)
            recipe = recipe_constructor(config)
            recipe.setup(config)
            recipe.train()
            recipe.cleanup()
