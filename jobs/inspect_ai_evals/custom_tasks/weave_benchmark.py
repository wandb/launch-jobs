"""
Weave dataset-based benchmark loader.

Reads configuration from run_config to create Inspect AI Tasks using Weave datasets.

Supports both multiple choice and open-ended (free-form) questions.
"""

from typing import Any, Iterator, Optional

import weave
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import accuracy, choice, exact, includes, match, model_graded_qa
from inspect_ai.solver import generate, multiple_choice


def _get_choices(row: dict, config: dict) -> Optional[list[str]]:
    """
    Get choices from the dataset row.
    choices_field: Field name to extract choices from dataset
    
    Returns None if task_type is "open_ended" or choices_field is not provided.
    """
    task_type = config.get("task_type", "multiple_choice")
    if task_type == "open_ended":
        return None
    
    choices_field = config.get("choices_field")
    if choices_field is None:
        raise ValueError("'choices_field' must be provided for multiple_choice tasks")
    return row[choices_field]


def _resolve_answer(row: dict, config: dict, choices: Optional[list[str]] = None) -> str:
    """
    Convert answer field to appropriate format.
    
    For multiple_choice: Convert to letter format (A, B, C, D, ...).
    For open_ended: Return the answer as-is (string).
    """
    answer_field = config["answer_field"]
    answer_value = row[answer_field]
    task_type = config.get("task_type", "multiple_choice")
    
    # For open-ended questions, return the answer as string directly
    if task_type == "open_ended":
        return str(answer_value)
    
    # For multiple choice, convert to letter format
    answer_format = config.get("answer_format", "index_0")
    
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
        if choices is None:
            raise ValueError("choices required for 'text' answer_format")
        try:
            idx = choices.index(answer_value)
            return chr(ord('A') + idx)
        except ValueError:
            raise ValueError(f"Answer '{answer_value}' not found in choices: {choices}")
    else:
        raise ValueError(f"Unknown answer_format: {answer_format}")


def _row_to_dict(row: Any) -> dict:
    """
    Convert a Weave row object to a dictionary.
    """
    if isinstance(row, dict):
        return row
    elif hasattr(row, "model_dump"):
        # Pydantic model support
        return row.model_dump()
    elif hasattr(row, "__dict__"):
        return vars(row)
    else:
        raise ValueError(f"Cannot convert row of type {type(row)} to dict")


def _load_weave_dataset(config: dict) -> list[dict]:
    """
    Load dataset from Weave.
    
    Supports:
    - Full URI: weave:///entity/project/object/name:version
    - Short ref: name or name:version (requires weave client to be initialized with entity/project)
    
    Args:
        config: Configuration dict with 'dataset' field containing Weave reference.
    
    Returns:
        List of row dictionaries from the Weave dataset.
    """
    dataset_ref = config["dataset"]
    
    try:
        dataset = weave.ref(dataset_ref).get()
    except Exception as e:
        raise ValueError(
            f"Failed to load Weave dataset '{dataset_ref}'. "
            f"Make sure the dataset exists and weave client is initialized. Error: {e}"
        )
    
    # Weave Dataset has 'rows' attribute
    if hasattr(dataset, "rows"):
        return [_row_to_dict(row) for row in dataset.rows]
    # If it's already a list-like object
    elif hasattr(dataset, "__iter__"):
        return [_row_to_dict(row) for row in dataset]
    else:
        raise ValueError(
            f"Weave object '{dataset_ref}' is not a valid dataset. "
            f"Expected a Dataset with 'rows' attribute or an iterable."
        )


def _iter_samples(config: dict, limit: Optional[int] = None) -> Iterator[Sample]:
    """
    Load Weave dataset according to config and generate Samples.
    """
    ds = _load_weave_dataset(config)
    
    question_field = config["question_field"]
    metadata_fields = config.get("metadata_fields", [])
    base_prompt = config.get("base_prompt", "")
    task_type = config.get("task_type", "multiple_choice")
    
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
        
        # For open_ended tasks, don't include choices
        if task_type == "open_ended":
            yield Sample(
                input=full_input,
                target=target,
                id=i,
                metadata=metadata if metadata else None,
            )
        else:
            yield Sample(
                input=full_input,
                choices=choices,
                target=target,
                id=i,
                metadata=metadata if metadata else None,
            )


def _get_solver(config: dict):
    """
    Get the appropriate solver based on task_type.
    """
    task_type = config.get("task_type", "multiple_choice")
    
    if task_type == "multiple_choice":
        return multiple_choice()
    elif task_type == "open_ended":
        return generate()
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Use 'multiple_choice' or 'open_ended'.")


def _get_scorer(config: dict):
    """
    Get the appropriate scorer based on task_type and scorer_type.
    
    For multiple_choice: defaults to "choice"
    For open_ended: defaults to "exact"
    
    Available scorer_types:
        - choice: For multiple choice (letter matching)
        - exact: Exact string match
        - includes: Check if target is included in response
        - match: Regex pattern matching (requires 'scorer_pattern' in config)
        - model_graded: LLM-based grading
    """
    task_type = config.get("task_type", "multiple_choice")
    
    # Set default scorer based on task_type
    if task_type == "multiple_choice":
        default_scorer = "choice"
    else:
        default_scorer = "exact"
    
    scorer_type = config.get("scorer_type", default_scorer)
    
    if scorer_type == "choice":
        return choice()
    elif scorer_type == "exact":
        return exact()
    elif scorer_type == "includes":
        return includes()
    elif scorer_type == "match":
        pattern = config.get("scorer_pattern", r".*")
        return match(pattern=pattern)
    elif scorer_type == "model_graded":
        return model_graded_qa()
    else:
        raise ValueError(
            f"Unknown scorer_type: {scorer_type}. "
            "Use 'choice', 'exact', 'includes', 'match', or 'model_graded'."
        )


def build_weave_task(
    config: dict[str, Any],
    limit: Optional[int] = None
) -> Task:
    """
    Create an Inspect AI Task based on Weave dataset configuration.
    
    Args:
        config: Benchmark configuration dictionary (required).
            Required fields:
                - dataset: Weave dataset reference 
                    (e.g., "weave:///entity/project/object/my-dataset:v1" or "my-dataset:latest")
                - question_field: Field name for question text
                - answer_field: Field name for correct answer
            Required for multiple_choice:
                - choices_field: Field name for answer choices
            Optional fields:
                - task_type: "multiple_choice" (default) or "open_ended"
                - scorer_type: "choice", "exact", "includes", "match", "model_graded"
                    (defaults: "choice" for multiple_choice, "exact" for open_ended)
                - scorer_pattern: Regex pattern for "match" scorer_type
                - answer_format: "index_0", "index_1", "letter", "text" (default: "index_0")
                    (only used for multiple_choice)
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
    
    Example config (multiple choice):
        {
            "dataset": "weave:///wandb-smle/eval-datasets/object/ifeval-ko:latest",
            "question_field": "question",
            "choices_field": "choices",
            "answer_field": "answer",
            "answer_format": "index_0",
            "task_type": "multiple_choice",
            "task_name": "ifeval_ko",
            "language": "ko"
        }
    
    Example config (open-ended / math):
        {
            "dataset": "weave:///wandb-smle/eval-datasets/object/math-problems:latest",
            "question_field": "problem",
            "answer_field": "solution",
            "task_type": "open_ended",
            "scorer_type": "exact",
            "task_name": "math_eval",
            "language": "en"
        }
    """
    if config is None:
        raise ValueError(
            "config is required. Please provide weave_benchmark configuration "
            "in your run config when using 'custom/weave' task."
        )
    
    task_type = config.get("task_type", "multiple_choice")
    
    # Validate required fields based on task_type
    required_fields = ["dataset", "question_field", "answer_field"]
    if task_type == "multiple_choice":
        required_fields.append("choices_field")
    
    missing_fields = [f for f in required_fields if f not in config]
    if missing_fields:
        raise ValueError(f"Missing required fields in weave_benchmark config: {missing_fields}")
    
    samples = list(_iter_samples(config, limit=limit))
    
    if not samples:
        raise ValueError(
            f"No samples loaded from Weave dataset '{config['dataset']}'. "
            "Check that the dataset exists and has data."
        )
    
    task_name = config.get("task_name", "weave_benchmark")
    
    return Task(
        name=task_name,
        dataset=samples,
        solver=_get_solver(config),
        scorer=_get_scorer(config),
        metrics=[accuracy()],
        metadata={
            "source": config.get("source", "weave"),
            "language": config.get("language", "unknown"),
            "dataset": config["dataset"],
            "task_type": task_type,
        },
    )


