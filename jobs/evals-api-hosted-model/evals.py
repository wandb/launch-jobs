import wandb
import subprocess
import os
import weave
from weave import Evaluation
import asyncio
import openai
from pydantic import BaseModel, Field
import pandas as pd
from wandb.sdk import launch
from weave.flow import leaderboard
from weave.trace.ref_util import get_ref
import time
import requests
from requests.exceptions import RequestException


class Answer(BaseModel):
    answer: str


class StepByStepAnswer(BaseModel):
    reasoning: str = Field(description="Step by step reasoning")
    answer: str = Field(description="The final answer and nothing else")


class ChatModel(weave.Model):
    base_url: str
    model: str
    api_key: str

    @weave.op
    def predict(self, system_prompt: str, user_prompt: str):
        client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)

        response = client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"{system_prompt}.",
                },
                {"role": "user", "content": user_prompt},
            ],
            response_format=Answer,
        )

        return response.choices[0].message.parsed


class StepByStepChatModel(weave.Model):
    base_url: str
    model: str
    api_key: str

    @weave.op
    def predict(self, system_prompt: str, user_prompt: str):
        client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)

        response = client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"{system_prompt}. Provide step by step reasoning before answering.",
                },
                {"role": "user", "content": user_prompt},
            ],
            response_format=StepByStepAnswer,
        )

        return response.choices[0].message.parsed


@weave.op
def exact_match(expected_answer: str, output: StepByStepAnswer) -> bool:
    return expected_answer == output.answer


def hf_dataset(options: dict) -> weave.Dataset:
    name = options["name"]
    path = options["path"]
    limit = options.get("limit", None)
    mapping = options["mapping"]
    parq = pd.read_parquet(path)
    if limit is not None:
        parq = parq.head(limit)
        name = f"{name}-lim{limit}"
    rows = [mapping(row) for _, row in parq.iterrows()]
    ds = weave.Dataset(name=name, rows=rows)
    return ds


mmlu_evaluation = Evaluation(
    name="mmlu",
    dataset=hf_dataset(
        {
            "name": "mmlu",
            "limit": 10,
            "path": "hf://datasets/cais/mmlu/all/test-00000-of-00001.parquet",
            "mapping": lambda example: {
                "system_prompt": "Answer the question by providing a single letter (A, B, C, or D) that corresponds to the correct answer.",
                "user_prompt": f"""Question: {example["question"]}
Choices:
{
    chr(10).join(
        [f"{chr(ord('A')+i)}. {choice}" for i, choice in enumerate(example["choices"])]
    )
}
""",
                "expected_answer": chr(ord("A") + example["answer"]),
            },
        }
    ),
    scorers=[exact_match],
)


aime2025_evaluation = Evaluation(
    name="aime2025",
    dataset=hf_dataset(
        {
            "name": "aime2025",
            "limit": 10,
            "path": "hf://datasets/yentinglin/aime_2025/data/train-00000-of-00001-243207c6c994e1bd.parquet",
            "mapping": lambda example: {
                "system_prompt": "Solve the problem, providing the correct integer result.",
                "user_prompt": example["problem"],
                "expected_answer": example["answer"],
            },
        }
    ),
    scorers=[exact_match],
)

job_input_schema = {
    "type": "object",
    "properties": {
        "base_url": {
            "type": "string",
            "description": "Base URL of the API",
            # "default": "https://api.openai.com/v1",
        },
        "model": {
            "type": "string",
            "description": "Model to use",
            # "default": "gpt-4.1-nano-2025-04-14",
        },
        "step_by_step": {
            "type": "boolean",
            "description": 'Add "Provide step by step reasoning before answering." to the system prompt',
            # "default": True,
        },
        "api_key": {
            "type": "string",
            "description": "API key to use",
        },
    },
}


def prefix_dict_keys(d: dict, prefix: str) -> dict:
    """Prefix all keys in a dictionary with prefix + '/' + original key."""
    return {f"{prefix}/{k}": v for k, v in d.items()}


with wandb.init(
    settings=wandb.Settings(
        project="job", disable_job_creation=False, job_source="image", job_name="Evaluate API-hosted Model"
    )
) as run:
    launch.manage_wandb_config(
        include=["base_url", "model", "api_key"],
        schema=job_input_schema,
    )

    # TODO: what is the right way to set defaults?
    base_url = run.config.get("base_url", "https://api.openai.com/v1")
    model_name = run.config.get("model", "gpt-4.1-nano-2025-04-14")
    step_by_step = run.config.get("step_by_step", False)
    api_key = run.config.get("api_key")


    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY is not set")

    # Start tracking the evaluation
    weave.init(run.project)

    if step_by_step:
        model = StepByStepChatModel(
            base_url=base_url,
            model=model_name,
            api_key=api_key,
        )
        model.name = f"{model_name}-stepbystep"
    else:
        model = ChatModel(
            base_url=base_url,
            model=model_name,
            api_key=api_key,
        )
        model.name = model_name

    run.name = model.name

    try:
        # Run the evaluation
        mmlu_results = asyncio.run(
            mmlu_evaluation.evaluate(
                model, __weave={"display_name": f"mmlu-{model.name}"}
            )
        )
        mmlu_results_prefixed = prefix_dict_keys(mmlu_results, "mmlu")

        aime2025_results = asyncio.run(
            aime2025_evaluation.evaluate(
                model, __weave={"display_name": f"aime2025-{model.name}"}
            )
        )
        aime2025_results_prefixed = prefix_dict_keys(aime2025_results, "aime2025")

        # Combine results and log in a single call
        combined_results = {**mmlu_results_prefixed, **aime2025_results_prefixed}
        run.log(combined_results)
    finally:
        pass

    spec = leaderboard.Leaderboard(
        name="leaders",
        description="The crème de la crème",
        columns=[
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=get_ref(mmlu_evaluation).uri(),
                scorer_name="exact_match",
                summary_metric_path="true_fraction",
            ),
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=get_ref(aime2025_evaluation).uri(),
                scorer_name="exact_match",
                summary_metric_path="true_fraction",
            ),
        ],
    )

    ref = weave.publish(spec)