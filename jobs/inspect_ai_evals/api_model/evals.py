import os

import wandb
# import weave
from inspect_ai import eval_set
from inspect_ai._eval.loader import load_tasks
from inspect_ai._eval.task import task_with
from inspect_ai.model import get_model
from inspect_ai.util import SandboxEnvironmentSpec
from inspect_ai.util._sandbox.registry import registry_find_sandboxenv
# from wandb.sdk import launch
# from weave.evaluation.eval_imperative import _active_evaluation_loggers

# from leaderboard import create_leaderboard
# from launch_secrets import get_launch_secret_from_env
from datasets.exceptions import DatasetNotFoundError

# Import to register the Aviato sandbox environment
import inspect_aviato_sandbox  # noqa: F401

# Use inspect_evals swe_bench with Aviato support
from inspect_evals.swe_bench import swe_bench
from inspect_evals.swe_bench.swe_bench_tasks import swe_bench_react_agent

INSPECT_EVAL_PREFIX = "inspect_evals/"


def create_aviato_sandbox_spec_with_env(
    sandbox_type: str,
    image_name: str,
    allow_internet: bool,
) -> SandboxEnvironmentSpec:
    """Create Aviato sandbox spec with environment variables."""
    sandbox_cls = registry_find_sandboxenv("aviato")
    
    config = {
        "container_image": image_name,
        "tags": (),
        "base_url": os.environ.get("AVIATO_BASE_URL", "https://atc.cwaviato.com"),
        "environment_variables": {
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
            "WANDB_ENTITY_NAME": os.environ.get("WANDB_ENTITY_NAME", ""),
            "WANDB_PROJECT_NAME": os.environ.get("WANDB_PROJECT_NAME", ""),
            # Model API keys
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
            # Hugging Face API keys
            "HUGGINGFACE_TOKEN": os.environ.get("HUGGINGFACE_TOKEN", ""),
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            "HUGGINGFACE_HUB_TOKEN": os.environ.get("HUGGINGFACE_HUB_TOKEN", ""),
            # Models
            "INSPECT_EVAL_MODEL": os.environ.get("INSPECT_EVAL_MODEL", ""),
            "INSPECT_GRADER_MODEL": os.environ.get("INSPECT_GRADER_MODEL", ""),
        },
    }
    
    typed_config = sandbox_cls.config_deserialize(config)
    return SandboxEnvironmentSpec(type="aviato", config=typed_config)


def get_native_providers() -> set[str]:
    try:
        # Import the providers module to trigger registration
        from inspect_ai.model._providers import providers  # noqa: F401
        from inspect_ai._util.registry import registry_find, registry_info

        # Find all registered modelapi items
        modelapis = registry_find(lambda info: info.type == "modelapi")

        # Extract provider names (strip the "inspect_ai/" prefix)
        names = set()
        for api in modelapis:
            try:
                info = registry_info(api)
                provider_name = info.name.replace("inspect_ai/", "")
                # Exclude internal/test providers
                if provider_name not in ["mockllm", "none"]:
                    names.add(provider_name)
            except Exception:
                continue

        if names:
            return names
    except (ImportError, AttributeError):
        pass

    # Fallback to hardcoded list from https://inspect.aisi.org.uk/providers.html
    return {
        "anthropic",
        "azureai",
        "bedrock",
        "cf",
        "fireworks",
        "google",
        "groq",
        "grok",
        "hf",
        "hf-inference-providers",
        "llama-cpp-python",
        "mistral",
        "ollama",
        "openai",
        "openai-api",
        "openrouter",
        "perplexity",
        "sambanova",
        "sglang",
        "together",
        "transformer_lens",
        "vllm",
    }


INSPECT_NATIVE_PROVIDERS = get_native_providers()


def resolve_model_name(model_name: str):
    parts = model_name.split("/", 1)
    if len(parts) < 2:
        raise ValueError("Hint: Model name must be in the format 'provider/model-name'")
    provider = parts[0].lower()
    if provider in INSPECT_NATIVE_PROVIDERS:
        return model_name
    return f"openai-api/{model_name}"


def main():
    # with wandb.init(entity="wandb", project="launch-tutorial") as run:
    # os.environ.setdefault("WANDB_ENTITY", run.entity)
    # os.environ.setdefault("WANDB_PROJECT", run.project)
    # weave_client = weave.init(f"{run.entity}/{run.project}")

    hf_token = ""
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
    os.environ.setdefault("HF_TOKEN", hf_token)
    os.environ.setdefault("HUGGINGFACE_TOKEN", hf_token)

    api_key_name = "OPENAI_API_KEY"
    api_key = ""
    os.environ.setdefault(api_key_name, api_key)

    try:
        model_name = "openai/gpt-4o"
        base_url = "https://api.openai.com/v1"
        resolved_model_name = resolve_model_name(model_name)
        model = get_model(resolved_model_name, api_key=api_key, base_url=base_url)
        os.environ.setdefault("INSPECT_EVAL_MODEL", resolved_model_name)
        os.environ.setdefault("INSPECT_GRADER_MODEL", resolved_model_name)
    except Exception as e:
        # weave_client.finish()
        raise e

    # failed_tasks = []
    # for task in ["swe_bench"]:
    try:
        loaded_task = [
            task_with(
                swe_bench(
                    dataset="princeton-nlp/SWE-bench_Lite",
                    split="test",
                    instance_ids=[
                        "astropy__astropy-12907",
                        # "astropy__astropy-14182",
                        # "astropy__astropy-14365",
                        # "astropy__astropy-14995",
                        # "astropy__astropy-6938",
                        # "astropy__astropy-7746",
                        # "django__django-10914",
                        # "django__django-10924",
                        # "django__django-11001",
                        # "django__django-11019",
                    ],
                    sandbox_type="aviato",
                    # sandbox_type="docker",
                    sandbox_config=create_aviato_sandbox_spec_with_env,
                    arch="x86_64",
                    solver=swe_bench_react_agent(),
                ),
                model=model
            )
        ]
        sample_limit = 5
        success, eval_logs = eval_set(
            tasks=loaded_task,
            log_dir="logs/",
            limit=sample_limit
            if sample_limit
            else None,  # evaluate all samples if sample_limit is set to 0
            retry_attempts=1,
            retry_wait=10,
            log_dir_allow_dirty=True,
        )

        print("eval logs", eval_logs)
        if not success:
            raise Exception("Task did not run successfully")
            # wandb.termerror(f"Task {task} did not run successfully")
            # failed_tasks.append(
            #     (
            #         task,
            #         Exception(
            #             "Task did not complete successfully. Check the logs for more details."
            #         ),
            #     )
            # )
            # continue

        # create_leaderboard()
        # for eval_logger in _active_evaluation_loggers:
        #     print("eval logger", eval_logger)
        #     print("eval logger attributes", dir(eval_logger))

    except Exception as e:
        # wandb.termerror(f"Task {task} failed to run")
        # failed_tasks.append((task, e))
        raise e

    # if failed_tasks:
    #     wandb.termerror(
    #         f"The following tasks failed to run: {[task for (task, _) in failed_tasks]}"
    #     )
    #     for task, e in failed_tasks:
    #         wandb.termerror(f"Task {task} failed to run with error: {e}")
    #         if isinstance(e, DatasetNotFoundError):
    #             wandb.termlog(
    #                 "Hint: This may be a gated dataset. Please check that you have set the 'Hugging Face Token' in the job input and have accepted the agreement on Hugging Face."
    #             )
    #     run.finish(exit_code=1)

    # weave_client.finish()


if __name__ == "__main__":
    main()
