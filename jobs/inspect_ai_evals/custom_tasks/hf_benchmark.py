"""
일반화된 Hugging Face 데이터셋 기반 벤치마크 로더.

run_config 또는 custom_benchmark.yaml 설정 파일을 읽어서 Inspect AI Task를 생성합니다.
"""

import os
from typing import Any, Optional

import yaml
from datasets import load_dataset
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import accuracy, choice
from inspect_ai.solver import multiple_choice


def load_benchmark_config(config_path: Optional[str] = None) -> dict[str, Any]:
    """
    벤치마크 설정 파일을 로드합니다.
    
    Args:
        config_path: 설정 파일 경로. None이면 기본 경로 사용.
    
    Returns:
        설정 딕셔너리
    """
    if config_path is None:
        # 기본 경로: 이 파일과 같은 디렉토리의 custom_benchmark.yaml
        config_path = os.path.join(
            os.path.dirname(__file__), "custom_benchmark.yaml"
        )
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_choices(row: dict, config: dict) -> list[str]:
    """
    선택지를 가져옵니다.
    choices_field: 데이터셋에서 가져올 필드명
    """
    choices_field = config.get("choices_field")
    if choices_field is None:
        raise ValueError("'choices_field' must be provided")
    return row[choices_field]


def _resolve_answer(row: dict, config: dict, choices: list[str]) -> str:
    """
    정답 필드를 letter (A, B, C, D, ...) 형식으로 변환합니다.
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
        # 이미 letter 형식 (A, B, C, D)
        return str(answer_value).upper()
    elif answer_format == "text":
        # 텍스트 정답 -> choices에서 찾아서 letter로 변환
        try:
            idx = choices.index(answer_value)
            return chr(ord('A') + idx)
        except ValueError:
            raise ValueError(f"Answer '{answer_value}' not found in choices: {choices}")
    else:
        raise ValueError(f"Unknown answer_format: {answer_format}")


def _iter_samples(config: dict, limit: Optional[int] = None):
    """
    설정에 따라 데이터셋을 로드하고 Sample을 생성합니다.
    """
    # 데이터셋 로드
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
        
        # base_prompt가 있으면 질문 앞에 추가
        full_input = f"{base_prompt}{question}" if base_prompt else question
        
        # 메타데이터 수집
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


def build_custom_task(
    config: Optional[dict[str, Any]] = None,
    config_path: Optional[str] = None,
    limit: Optional[int] = None
) -> Task:
    """
    설정을 기반으로 Inspect AI Task를 생성합니다.
    
    Args:
        config: 벤치마크 설정 딕셔너리. 제공되면 config_path보다 우선.
        config_path: 설정 파일 경로. config가 None이고 이것도 None이면 기본 custom_benchmark.yaml 사용.
        limit: 샘플 수 제한 (0 또는 None이면 전체 사용).
    
    Returns:
        Inspect AI Task
    """
    # config가 직접 제공되면 사용, 아니면 파일에서 로드
    if config is None:
        config = load_benchmark_config(config_path)
    
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
