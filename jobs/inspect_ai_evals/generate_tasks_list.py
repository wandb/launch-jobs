import importlib.resources

import yaml
import json


def load_listing_yaml():
    import inspect_evals  # the root module

    with (
        importlib.resources.files(inspect_evals).joinpath("listing.yaml").open("r") as f
    ):
        return yaml.safe_load(f)


def generate_task_list():
    listing = load_listing_yaml()
    tasks = [
        f"inspect_evals/{task['name']}"
        for item in listing
        for task in item.get("tasks", [])
        if "sandbox" not in item.get("metadata", {}) and task.get("name")
    ]

    unsupported_tasks = [
        # Sandbox tasks
        "inspect_evals/agent_bench_os",
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
        "inspect_evals/mind2web_sc",
        "inspect_evals/threecb",
        # tasks return empty on load_tasks()
        "inspect_evals/make-me-say",
        "inspect_evals/mind2web",
        "inspect_evals/sciknoweval",
        "inspect_evals/Zerobench",
        "inspect_evals/Zerobench Subquestions",
        "inspect_evals/personality_PRIME",
        # unable to set scorer via environment variables
        "inspect_evals/fortress_benign",
        "inspect_evals/fortress_adversarial",
        # running into an auth error when running task
        "inspect_evals/abstention_bench",
        # Skipping for now. Requires additional input for the judge model
        "inspect_evals/writingbench",
        # Pins pandas version to 2.2.2
        "inspect_evals/livebench",
    ]

    tasks = [
        task.replace("inspect_evals/", "")
        for task in tasks
        if task not in unsupported_tasks
    ]

    tasks = sorted(tasks)

    return tasks


if __name__ == "__main__":
    tasks = generate_task_list()
    print(json.dumps(tasks, indent=2))
