"""Run Inspect AI evaluations.

Thin wrapper around ``inspect_ai.eval_set()`` that adds:
- W&B run tracking (``wandb.init`` + ``weave.init``)
- Model name resolution (auto-detect provider, prepend ``openai-api/`` for non-native)
- Aviato sandbox (always)
- Optional Weave leaderboard publishing

CLI args mirror ``inspect eval`` where possible.  Infrastructure knobs
(retries, concurrency, log dir) are hardcoded to sensible defaults.
"""

from __future__ import annotations

import argparse
import os
import sys

import wandb
import weave
from inspect_ai import eval_set

import inspect_aviato_sandbox  # noqa: F401  # registers aviato sandbox environment

from datasets.exceptions import DatasetNotFoundError
from leaderboard import create_leaderboard

INSPECT_EVAL_PREFIX = "inspect_evals/"


def resolve_api_keys(resolved_model: str) -> None:
    """Map generic secret env vars to the provider-specific names inspect_ai expects.

    The sandbox executor passes secrets as env vars keyed by their config
    field name (e.g. ``model_api_key``).  This helper re-maps them to the
    env vars each provider SDK actually reads.
    """
    # MODEL_API_KEY → provider-specific env var that inspect_ai expects.
    # Native:     "provider/model"             → PROVIDER_API_KEY
    # Non-native: "openai-api/provider/model"  → PROVIDER_API_KEY
    model_api_key = os.environ.get("MODEL_API_KEY")
    if model_api_key:
        provider, *_ = resolved_model.split("/")
        if provider == "openai-api":
            _, provider, *_ = resolved_model.split("/")
        env_var = f"{provider.upper().replace('-', '_')}_API_KEY"
        os.environ.setdefault(env_var, model_api_key)

    # SCORER_API_KEY → OPENAI_API_KEY (default scorer is OpenAI)
    scorer_api_key = os.environ.get("SCORER_API_KEY")
    if scorer_api_key:
        os.environ.setdefault("OPENAI_API_KEY", scorer_api_key)
        os.environ.setdefault("AZURE_OPENAI_API_KEY", scorer_api_key)

    # HF_TOKEN → HuggingFace env vars (for gated datasets)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

# Infrastructure defaults — users don't need to think about these.
RETRY_ATTEMPTS = 3
RETRY_WAIT = 10
MAX_TASKS = 10


def get_native_providers() -> set[str]:
    try:
        from inspect_ai.model._providers import providers  # noqa: F401
        from inspect_ai._util.registry import registry_find, registry_info

        modelapis = registry_find(lambda info: info.type == "modelapi")

        names = set()
        for api in modelapis:
            try:
                info = registry_info(api)
                provider_name = info.name.replace("inspect_ai/", "")
                if provider_name not in ("mockllm", "none"):
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


def resolve_model_name(model_name: str) -> str:
    parts = model_name.split("/", 1)
    if len(parts) < 2:
        raise ValueError("Hint: Model name must be in the format 'provider/model-name'")
    provider = parts[0].lower()
    if provider in INSPECT_NATIVE_PROVIDERS:
        return model_name
    return f"openai-api/{model_name}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Inspect AI evaluations")
    parser.add_argument(
        "tasks", nargs="+",
        help="Task names (e.g. swebench mmlu_pro)",
    )
    parser.add_argument(
        "-m", "--model", required=True,
        help="Model name in provider/model format (e.g. openai/gpt-4o)",
    )
    parser.add_argument(
        "--model-base-url", default=None,
        help="API base URL for the model endpoint",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max samples per task (0 or omitted = all)",
    )
    parser.add_argument(
        "--create-leaderboard", action="store_true", default=False,
        help="Publish results to Weave leaderboard",
    )
    return parser.parse_args(argv)


def setup_env(model_name: str, base_url: str | None = None) -> str:
    """Resolve model name and set env vars for inspect_ai.

    Returns the resolved model name.
    """
    resolved = resolve_model_name(model_name)

    if base_url:
        provider = model_name.split("/", 1)[0]
        env_var = f"{provider.upper().replace('-', '_')}_BASE_URL"
        os.environ.setdefault(env_var, base_url)

    os.environ.setdefault("INSPECT_EVAL_MODEL", resolved)
    resolve_api_keys(resolved)
    return resolved


def run_eval(args: argparse.Namespace) -> int:
    """Run Inspect AI evaluations. Returns 0 on success, 1 on failure."""
    try:
        resolved = setup_env(args.model, args.model_base_url)
    except ValueError as e:
        wandb.termerror(str(e))
        return 1

    task_names = [
        t if "/" in t else f"{INSPECT_EVAL_PREFIX}{t}" for t in args.tasks
    ]

    with wandb.init(config={"model": args.model, "tasks": args.tasks}) as run:
        weave_client = weave.init(f"{run.entity}/{run.project}")

        try:
            success, logs = eval_set(
                tasks=task_names,
                model=resolved,
                model_base_url=args.model_base_url,
                sandbox="aviato",
                limit=args.limit or None,
                log_dir="logs/",
                log_dir_allow_dirty=True,
                retry_attempts=RETRY_ATTEMPTS,
                retry_wait=RETRY_WAIT,
                max_tasks=MAX_TASKS,
            )
        except DatasetNotFoundError as e:
            wandb.termerror(f"Evaluation failed: {e}")
            wandb.termlog(
                "Hint: This may be a gated dataset. Please check that you have "
                "set the 'Hugging Face Token' in the job input and have accepted "
                "the agreement on Hugging Face."
            )
            run.finish(exit_code=1)
            weave_client.finish()
            return 1
        except Exception as e:
            wandb.termerror(f"Evaluation failed: {e}")
            wandb.termlog(
                "Hint: Please check that the model name and API key are correct."
            )
            run.finish(exit_code=1)
            weave_client.finish()
            return 1

        if not success:
            for log in logs:
                if log.status != "success":
                    wandb.termerror(f"Task {log.eval.task}: {log.status}")
                    if log.error:
                        wandb.termerror(log.error.message)
            run.finish(exit_code=1)
            weave_client.finish()
            return 1

        if args.create_leaderboard:
            try:
                create_leaderboard()
            except Exception as e:
                wandb.termerror(f"Leaderboard publishing failed: {e}")

        run.finish(exit_code=0)
        weave_client.finish()
        return 0


def main():
    args = parse_args()
    sys.exit(run_eval(args))


if __name__ == "__main__":
    main()
