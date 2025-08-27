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
                "description": "Select the tasks to run",
                "items": {
                    "type": "string",
                    "enum": tasks,
                },
            },
            "limit": {
                "type": "integer",
                "title": "Limit",
                "description": "Maximum number of samples to run in a job across all tasks",
            },
            "model": {
                "type": "object",
                "title": "Model",
                "description": "Model to use for the task",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "title": "Model",
                        "description": "Name of the model to use for the task",
                        "enum": ["openai/gpt-4.1-mini", "openai/gpt-4.1-nano"],
                    },
                    "base_url": {
                        "type": "string",
                        "title": "Base URL",
                        "description": "Base URL for the model API",
                    },
                    "api_key_var": {
                        "type": "string",
                        "title": "API Key Variable",
                        "description": (
                            "Environment variables that contain the API key for the model"
                        ),
                        # "format": "secret",
                    },
                },
            },
            "create_leaderboard": {
                "type": "boolean",
                "title": "Create Leaderboard",
                "description": "Create a leaderboard from eval loggers",
            },
        },
    }


if __name__ == "__main__":
    import json

    schema = generate_schema()
    print(json.dumps(schema, indent=2))
