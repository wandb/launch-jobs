import importlib.resources

import yaml


def load_listing_yaml():
    import inspect_evals  # the root module

    with (
        importlib.resources.files(inspect_evals).joinpath("listing.yaml").open("r") as f
    ):
        return yaml.safe_load(f)


def generate_schema():
    listing = load_listing_yaml()
    tasks = [
        f"inspect_evals/{task['name']}"
        for item in listing
        for task in item.get("tasks", [])
        if "sandbox" not in item.get("metadata", {}) and task.get("name")
    ]
    
    unsupported_tasks = [
        # Sandbox tasks
        "inspect_evals/cybench",
        "inspect_evals/gdm_intercode_ctf",
        "inspect_evals/gdm_approved_directories",
        "inspect_evals/gdm_calculator_improvement",
        "inspect_evals/gdm_context_length_mod_instrumental_only",
        "inspect_evals/gdm_context_length_mod_irreversibility_only",
        "inspect_evals/gdm_database_tool",
        "inspect_evals/gdm_latency_calculator",
        "inspect_evals/gdm_max_messages_calculator",
        "inspect_evals/gdm_max_tokens",
        "inspect_evals/gdm_oversight_frequency",
        "inspect_evals/gdm_read_logs",
        "inspect_evals/gdm_turn_off_filters",
        "inspect_evals/gdm_classifier_evasion",
        "inspect_evals/gdm_cover_your_tracks",
        "inspect_evals/gdm_oversight_pattern",
        "inspect_evals/gdm_strategic_rule_breaking",
        
        # Skipping for now. Memory heavy and can lead to crashes.
        "inspect_evals/docvqa",
        "inspect_evals/bold",
        
        # Requires further investigation
        "inspect_evals/mathvista",
        "inspect_evals/squad",
        "inspect_evals/mind2web",
        "inspect_evals/sciknoweval",
        "inspect_evals/Zerobench",
        "inspect_evals/Zerobench Subquestions",
        "inspect_evals/personality_PRIME",
        
        # Skipping for now. Requires additional input for the judge model
        "inspect_evals/writingbench",
        
        # Pins pandas version to 2.2.2
        "inspect_evals/livebench",
    ]

    tasks = [task for task in tasks if task not in unsupported_tasks]
    
    tasks = sorted(tasks)

    return {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "title": "Select tasks",
                "description": "Select the tasks to run. For more information on the tasks, see the docs at: https://inspect.aisi.org.uk/evals/.",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "string",
                    "enum": tasks,
                },
            },
            "limit": {
                "type": "integer",
                "title": "Sample limit",
                "description": "Maximum number of samples to evaluate for each task (e.g. 5, 10, 25, 50)",
            },
            "model": {
                "type": "object",
                "title": "Select model",
                "description": "Model to use for the evaluation job",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "title": "Model name",
                        "description": "Name of the model to use for the evaluation job",
                        "enum": [
                            "openai/chatgpt-4o-latest",
                            "openai/gpt-3.5-turbo",
                            "openai/gpt-4",
                            "openai/gpt-4-turbo",
                            "openai/gpt-4.1",
                            "openai/gpt-4.1-mini",
                            "openai/gpt-4.1-nano",
                            "openai/gpt-4o",
                            "openai/gpt-4o-mini",
                            "openai/gpt-5",
                            "openai/gpt-5-chat-latest",
                            "openai/gpt-5-mini",
                            "openai/gpt-5-nano",
                            "openai/gpt-audio",
                            "openai/o1",
                            "openai/o1-mini",
                            "openai/o1-pro",
                            "openai/o3",
                            "openai/o3-mini",
                            "openai/o3-pro",
                            "openai/o4-mini",
                            "anthropic/claude-opus-4-1",
                            "anthropic/claude-opus-4-0",
                            "anthropic/claude-sonnet-4-0",
                            "anthropic/claude-3-7-sonnet-latest",
                            "anthropic/claude-3-5-haiku-latest",
                            "google/gemini-2.5-pro",
                            "google/gemini-2.5-flash",
                            "google/gemini-2.0-flash-001",
                            "google/gemini-2.0-flash-lite-001",
                        ]
                    },
                    "api_key_var": {
                        "type": "string",
                        "title": "API Key",
                        "description": "API key for model access",
                        "format": "secret",
                    },
                },
            },
            "create_leaderboard": {
                "type": "boolean",
                "title": "Publish results to leaderboard",
                "description": "If enabled, the evaluation results will be published to your project's default 'Inspect AI Leaderboard'.",
            },
            "hf_token": {
                "type": "string",
                "title": "Hugging face token",
                "description": (
                    "(Optional) Personal access token used to read gated datasets from Hugging Face."
                ),
                "format": "secret",
            },
            "scorer_api_key": {
                "type": "string",
                "title": "Scorer API key",
                "description": "(Optional) Some evals use an OpenAI model as the default scorer.",
                "format": "secret",
            },
        },
    }


if __name__ == "__main__":
    import json

    schema = generate_schema()
    print(json.dumps(schema, indent=2))
