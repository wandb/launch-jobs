"""
KMMLU-Pro 한국어 라이선스 시험 벤치마크를 Inspect AI Task로 변환.

요구사항:
- HF_TOKEN 환경변수에 Hugging Face 액세스 토큰이 있어야 게이트 데이터에 접근 가능.
- 데이터 split은 "test"만 존재함.
"""

from typing import Optional

from datasets import load_dataset
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import accuracy, choice
from inspect_ai.solver import multiple_choice


def _iter_samples(limit: Optional[int] = None):
    ds = load_dataset("LGAI-EXAONE/KMMLU-Pro", split="test", streaming=False)
    for i, row in enumerate(ds):
        if limit and limit > 0 and i >= limit:
            break
        question = row["question"]
        options = row["options"]
        # dataset uses 1-based string index for the correct option
        sol_idx = int(row["solution"]) - 1
        # choice() scorer expects letter target (A, B, C, D, ...)
        target = chr(ord('A') + sol_idx)
        yield Sample(
            input=question,
            choices=options,
            target=target,
            id=i,
            metadata={
                "license_name": row.get("license_name"),
                "subject": row.get("subject"),
                "year": row.get("year"),
                "round": row.get("round"),
                "session": row.get("session"),
            },
        )


def build_kmmlu_task(task_name: str, limit: Optional[int] = None) -> Task:
    """
    task_name: currently supports "kmmlu_pro" (with or without custom/ prefix handled upstream).
    limit: 샘플 수 제한 (0 또는 None이면 전체 사용).
    """
    if task_name not in {"kmmlu_pro", "custom/kmmlu_pro"}:
        raise ValueError(f"Unsupported local task: {task_name}")

    samples = list(_iter_samples(limit=limit))
    return Task(
        name="kmmlu_pro",
        dataset=samples,
        solver=multiple_choice(),
        scorer=choice(),
        metrics=[accuracy()],
        metadata={
            "source": "KMMLU-Pro",
            "language": "ko",
            "license": "CC-BY-NC-ND-4.0",
        },
    )

