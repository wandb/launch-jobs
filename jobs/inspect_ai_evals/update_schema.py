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
                "title": "Tasks",
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
                "title": "Sample Limit",
                "description": "Maximum number of samples to evaluate for each task (e.g. 5, 10, 25, 50)",
            },
            "model": {
                "type": "object",
                "title": "Model",
                "description": "Model to use for the evaluation job",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "title": "Name",
                        "description": "Name of the model to use for the evaluation job",
                        "enum": ["openai/gpt-4.1-mini", "openai/gpt-4.1-nano"],
                    },
                    "base_url": {
                        "type": "string",
                        "title": "Base URL",
                        "description": "Base URL for the model API (e.g. https://api.openai.com/v1)",
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
                "title": "Create a Leaderboard?",
                "description": "Choose to create a leaderboard from eval loggers. This will be updated with the results of the evaluation.",
            },
            "hf_token": {
                "type": "string",
                "title": "Hugging Face Token",
                "description": (
                    "(Optional) Personal access token used to read gated datasets from Hugging Face."
                ),
                "format": "secret",
            },
        },
    }


if __name__ == "__main__":
    import json

    schema = generate_schema()
    print(json.dumps(schema, indent=2))
