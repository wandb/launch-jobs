from omegaconf import DictConfig

from helpers import execute

MODEL = "mistralai/Mistral-7B-v0.1"
DATASET = "bcanfieldsherman/torchtune/alpaca:v0"
CONFIG = DictConfig(
    {
        "device": "cuda",
        "dtype": "bf16",
        "log_every_n_steps": None,
        "seed": None,
        "shuffle": False,
        "compile": False,
        "max_steps_per_epoch": None,
        "gradient_accumulation_steps": 4,
        "resume_from_checkpoint": False,
        "enable_activation_checkpointing": True,
        "epochs": 3,
        "batch_size": 4,
        "tokenizer": {
            "_component_": "torchtune.models.mistral.mistral_tokenizer",
            "path": "${model_dir}/tokenizer.model",
        },
        "dataset": {
            "_component_": "torchtune.datasets.instruct_dataset",
            "source": "json",
            "data_files": "${data_dir}/train.json",
            "split": "train",
            "template": "AlpacaInstructTemplate",
            "train_on_input": True,
        },
        "model": {
            "_component_": "torchtune.models.mistral.qlora_mistral_7b",
            "lora_attn_modules": ["q_proj", "k_proj", "v_proj"],
            "apply_lora_to_mlp": True,
            "apply_lora_to_output": False,
            "lora_rank": 64,
            "lora_alpha": 16,
        },
        "metric_logger": {
            "_component_": "torchtune.utils.metric_logging.WandBLogger",
            "log_dir": "${output_dir}",
        },
        "optimizer": {
            "_component_": "torch.optim.AdamW",
            "lr": 2e-5,
        },
        "lr_scheduler": {
            "_component_": "torchtune.modules.get_cosine_schedule_with_warmup",
            "num_warmup_steps": 100,
        },
        "loss": {
            "_component_": "torch.nn.CrossEntropyLoss",
        },
        "checkpointer": {
            "_component_": "torchtune.utils.FullModelHFCheckpointer",
            "checkpoint_dir": "${model_dir}",
            "checkpoint_files": [
                "pytorch_model-00001-of-00002.bin",
                "pytorch_model-00002-of-00002.bin",
            ],
            "recipe_checkpoint": None,
            "output_dir": "${output_dir}",
            "model_type": "MISTRAL",
        },
        "profiler": {
            "_component_": "torchtune.utils.profiler",
            "enabled": False,
            "output_dir": "${output_dir}",
        },
    }
)


def main():
    execute(
        MODEL,
        "lora_finetune_single_device",
        "LoRAFinetuneRecipeSingleDevice",
        CONFIG,
        DATASET,
    )


if __name__ == "__main__":
    main()
