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

    return {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "title": "Tasks",
                "description": "Select the tasks to run. For more information on the tasks, see the https://inspect.aisi.org.uk/evals/.",
                "items": {
                    "type": "string",
                    "enum": tasks,
                },
            },
            "limit": {
                "type": "integer",
                "title": "Sample Limit",
                "description": "Maximum number of samples to evaluate for each task",
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
                        "description": "Base URL for the model API",
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
            },
        },
    }


if __name__ == "__main__":
    import json

    schema = generate_schema()
    print(json.dumps(schema, indent=2))
