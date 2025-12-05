import os

import wandb
import weave
from inspect_ai import eval_set
from inspect_ai._eval.loader import load_tasks
from inspect_ai._eval.task import task_with
from custom_tasks.kmmlu_pro import build_kmmlu_task
from inspect_ai.model import get_model
from wandb.sdk import launch

from leaderboard import create_leaderboard
from launch_secrets import get_launch_secret_from_env
from datasets.exceptions import DatasetNotFoundError

INSPECT_EVAL_PREFIX = "inspect_evals/"
CUSTOM_PREFIX = "custom/"


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


def _resolve_task_name(task: str) -> str:
    """
    Normalize task name so both inspect_evals/*와 custom/*를 지원.
    - inspect_evals/로 시작하면 그대로 사용
    - custom/로 시작하면 접두사 제거
    - 그 외는 기본적으로 inspect_evals/를 붙임(기존 호환)
    """
    if task.startswith(INSPECT_EVAL_PREFIX):
        return task
    if task.startswith(CUSTOM_PREFIX):
        return task.replace(CUSTOM_PREFIX, "")
    return f"{INSPECT_EVAL_PREFIX}{task}"


def main():
    config = launch.load_wandb_config()
    with wandb.init(config=dict(config)) as run:
        weave_client = weave.init(f"{run.entity}/{run.project}")

        _, hf_token = get_launch_secret_from_env("hf_token", run.config)
        if hf_token:
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
            os.environ.setdefault("HF_TOKEN", hf_token)
            os.environ.setdefault("HUGGINGFACE_TOKEN", hf_token)

        api_key_name, api_key = get_launch_secret_from_env(
            "api_key_var", run.config.get("model", {})
        )
        if api_key_name and api_key:
            os.environ.setdefault(api_key_name, api_key)

        # Some evals use hosted models as the default scorer.
        # These could be OpenAI, Anthropic, or Google models.
        # * For OpenAI, the API key name is "OPENAI_API_KEY".
        # * For Anthropic, the API key name is "ANTHROPIC_API_KEY".
        # * For Google, the API key name is "GOOGLE_API_KEY".
        # Otherwise, we can try to use the model selected by the user (see INSPECT_GRADER_MODEL below)
        scorer_api_key_name, scorer_api_key = get_launch_secret_from_env(
            "scorer_api_key", run.config
        )
        if scorer_api_key_name and scorer_api_key:
            os.environ.setdefault(scorer_api_key_name, scorer_api_key)

        try:
            model_name = run.config.get("model", {}).get("model_name")
            base_url = run.config.get("model", {}).get("base_url", None)
            if not model_name:
                wandb.termerror("Hint: Model name is required")
                weave_client.finish()
                return

            try:
                resolved_model_name = resolve_model_name(model_name)
            except ValueError as e:
                wandb.termerror(f"Invalid model name: {model_name}. {e}")
                weave_client.finish()
                raise e

            if resolved_model_name.startswith("openai-api/"):
                provider = resolved_model_name.split("/")[1]
                os.environ.setdefault(
                    f"{provider.upper().replace('-', '_')}_BASE_URL", base_url
                )

            model = get_model(resolved_model_name, api_key=api_key, base_url=base_url)

            os.environ.setdefault("INSPECT_EVAL_MODEL", resolved_model_name)
            os.environ.setdefault("INSPECT_GRADER_MODEL", resolved_model_name)
        except Exception as e:
            wandb.termerror(f"Error initializing model: {e}")
            wandb.termlog(
                "Hint: Please check if the job inputs for the model is correct ('Name' and 'API Key')"
            )
            weave_client.finish()
            raise e

        failed_tasks = []
        for task in run.config.get("tasks", []):
            try:
                task_name = _resolve_task_name(task)
                if task.startswith(CUSTOM_PREFIX):
                    loaded_task = [build_kmmlu_task(task_name, limit=run.config.get("limit"))]
                else:
                    loaded = load_tasks([task_name])[0]
                    loaded_task = [task_with(loaded, model=model)]
                sample_limit = run.config.get("limit", None)
                success, _ = eval_set(
                    tasks=loaded_task,
                    log_dir="logs/",
                    limit=sample_limit
                    if sample_limit
                    else None,  # evaluate all samples if sample_limit is set to 0
                    retry_attempts=1,
                    retry_wait=10,
                    log_dir_allow_dirty=True,
                )

                if not success:
                    wandb.termerror(f"Task {task} did not run successfully")
                    failed_tasks.append(
                        (
                            task,
                            Exception(
                                "Task did not complete successfully. Check the logs for more details."
                            ),
                        )
                    )
                    continue

                if run.config.get("create_leaderboard", True):
                    create_leaderboard()

            except Exception as e:
                wandb.termerror(f"Task {task} failed to run")
                failed_tasks.append((task, e))

        if failed_tasks:
            wandb.termerror(
                f"The following tasks failed to run: {[task for (task, _) in failed_tasks]}"
            )
            for task, e in failed_tasks:
                wandb.termerror(f"Task {task} failed to run with error: {e}")
                if isinstance(e, DatasetNotFoundError):
                    wandb.termlog(
                        "Hint: This may be a gated dataset. Please check that you have set the 'Hugging Face Token' in the job input and have accepted the agreement on Hugging Face."
                    )
            run.finish(exit_code=1)

        weave_client.finish()


if __name__ == "__main__":
    main()
