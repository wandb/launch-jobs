"""
Generalized Hugging Face dataset-based benchmark loader.

Reads configuration from run_config to create Inspect AI Tasks.
"""

from typing import Any, Optional

from datasets import load_dataset
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import accuracy, choice
from inspect_ai.solver import multiple_choice


def _get_choices(row: dict, config: dict) -> list[str]:
    """
    Get choices from the dataset row.
    choices_field: Field name to extract choices from dataset
    """
    choices_field = config.get("choices_field")
    if choices_field is None:
        raise ValueError("'choices_field' must be provided")
    return row[choices_field]


def _resolve_answer(row: dict, config: dict, choices: list[str]) -> str:
    """
    Convert answer field to letter format (A, B, C, D, ...).
    """
    answer_field = config["answer_field"]
    answer_format = config.get("answer_format", "index_0")
    answer_value = row[answer_field]
    
    if answer_format == "index_0":
        # 0-based index (0, 1, 2, 3) -> (A, B, C, D)
        return chr(ord('A') + int(answer_value))
    elif answer_format == "index_1":
        # 1-based index (1, 2, 3, 4) -> (A, B, C, D)
        return chr(ord('A') + int(answer_value) - 1)
    elif answer_format == "letter":
        # Already in letter format (A, B, C, D)
        return str(answer_value).upper()
    elif answer_format == "text":
        # Text answer -> find in choices and convert to letter
        try:
            idx = choices.index(answer_value)
            return chr(ord('A') + idx)
        except ValueError:
            raise ValueError(f"Answer '{answer_value}' not found in choices: {choices}")
    else:
        raise ValueError(f"Unknown answer_format: {answer_format}")


def _iter_samples(config: dict, limit: Optional[int] = None):
    """
    Load dataset according to config and generate Samples.
    """
    # Load dataset
    load_kwargs = {
        "path": config["dataset"],
        "split": config.get("split", "test"),
        "streaming": False,
    }
    if "name" in config and config["name"]:
        load_kwargs["name"] = config["name"]
    
    ds = load_dataset(**load_kwargs)
    
    question_field = config["question_field"]
    metadata_fields = config.get("metadata_fields", [])
    base_prompt = config.get("base_prompt", "")
    
    for i, row in enumerate(ds):
        if limit and limit > 0 and i >= limit:
            break
        
        question = row[question_field]
        choices = _get_choices(row, config)
        target = _resolve_answer(row, config, choices)
        
        # Prepend base_prompt to question if provided
        full_input = f"{base_prompt}{question}" if base_prompt else question
        
        # Collect metadata
        metadata = {}
        for field in metadata_fields:
            if field in row:
                metadata[field] = row.get(field)
        
        yield Sample(
            input=full_input,
            choices=choices,
            target=target,
            id=i,
            metadata=metadata if metadata else None,
        )


def build_hf_task(
    config: dict[str, Any],
    limit: Optional[int] = None
) -> Task:
    """
    Create an Inspect AI Task based on configuration.
    
    Args:
        config: Benchmark configuration dictionary (required).
            Required fields:
                - dataset: HF dataset path (e.g., "LGAI-EXAONE/KMMLU-Pro")
                - question_field: Field name for question text
                - choices_field: Field name for answer choices
                - answer_field: Field name for correct answer
            Optional fields:
                - name: Dataset subset name
                - split: Data split (default: "test")
                - answer_format: "index_0", "index_1", "letter", "text" (default: "index_0")
                - base_prompt: Prompt prefix to prepend to questions
                - task_name: Display name for the task
                - language: Language code (e.g., "ko", "en")
                - source: Source name for metadata
                - metadata_fields: List of fields to include in sample metadata
        limit: Sample limit (0 or None means use all samples).
    
    Returns:
        Inspect AI Task
    
    Raises:
        ValueError: If config is None or missing required fields.
    """
    if config is None:
        raise ValueError(
            "config is required. Please provide custom_benchmark configuration "
            "in your run config when using 'custom/benchmark' task."
        )
    
    # Validate required fields
    required_fields = ["dataset", "question_field", "choices_field", "answer_field"]
    missing_fields = [f for f in required_fields if f not in config]
    if missing_fields:
        raise ValueError(f"Missing required fields in custom_benchmark config: {missing_fields}")
    
    samples = list(_iter_samples(config, limit=limit))
    
    task_name = config.get("task_name", "custom_benchmark")
    
    return Task(
        name=task_name,
        dataset=samples,
        solver=multiple_choice(),
        scorer=choice(),
        metrics=[accuracy()],
        metadata={
            "source": config.get("source", config["dataset"]),
            "language": config.get("language", "unknown"),
            "dataset": config["dataset"],
        },
    )

