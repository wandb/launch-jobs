"""Minimal BERT finetuning example using the Yelp review dataset.

This was constructed from the quickstart in the HuggingFace documentation to
illustrate how to create a finetuning workflow with W&B Launch. For the original 
examples, see:
    https://huggingface.co/docs/transformers/en/training

This script uses the Yelp review dataset from HuggingFace's datasets library
to train a BERT model on a small subset of the data. The model is then evaluated
on a small subset of the test data. The training and evaluation metrics are
logged to W&B.
"""

import evaluate
import numpy as np
import wandb
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from wandb.sdk import launch

wandb.require("core")  # Required to use launch.manage_config_file().
launch.manage_config_file("trainer_args.yaml")


def main():
    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # Take a small subset of the data.
    small_train_dataset = (
        dataset["train"]
        .shuffle()
        .select(range(4000))
        .map(tokenize_function, batched=True)
    )
    small_eval_dataset = (
        dataset["test"]
        .shuffle()
        .select(range(1000))
        .map(tokenize_function, batched=True)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-cased",
        num_labels=5,
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    with open("trainer_args.yaml") as f:
        training_args = TrainingArguments(**yaml.safe_load(f))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()


if __name__ == "__main__":
    main()
