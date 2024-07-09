import importlib.util
import site
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download
from omegaconf import DictConfig

import wandb


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


def monkeypatch_checkpointing(recipe_class):
    """Monkeypatch checkpointing in the recipe."""

    # Make a copy of the original save_checkpoint method.
    _original_save_checkpoint = recipe_class.save_checkpoint

    def save_checkpoint(self, epoch):
        """Save the checkpoint."""
        _original_save_checkpoint(self, epoch)
        ckpt_dir = Path(self._checkpointer._output_dir)
        # List all files matching *_{epoch}.pt in the checkpoint directory.
        files = [f for f in ckpt_dir.iterdir() if f.name.endswith(f"_{epoch}.pt")]
        artifact = wandb.Artifact(
            name=f"model_{epoch}",
            type="model",
            metadata={
                "seed": self.seed,
                "epochs_run": self.epochs_run,
                "total_epochs": self.total_epochs,
                "max_steps_per_epochs": self.max_steps_per_epoch,
            },
        )
        for file in files:
            artifact.add_file(file)
        wandb.log_artifact(artifact)

    recipe_class.save_checkpoint = save_checkpoint


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
            monkeypatch_checkpointing(recipe_constructor)
            recipe = recipe_constructor(config)
            recipe.setup(config)
            recipe.train()
            recipe.cleanup()
