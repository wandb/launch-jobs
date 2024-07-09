from omegaconf import DictConfig

from helpers import execute

MODEL = "meta-llama/Meta-Llama-3-8B"
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
        "gradient_accumulation_steps": 16,
        "resume_from_checkpoint": False,
        "enable_activation_checkpointing": True,
        "epochs": 3,
        "batch_size": 2,
        "tokenizer": {
            "_component_": "torchtune.models.llama3.llama3_tokenizer",
            "path": "${model_dir}/original/tokenizer.model",
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
            "_component_": "torchtune.models.llama3.qlora_llama3_8b",
            "lora_attn_modules": ["q_proj", "k_proj", "v_proj", "output_proj"],
            "apply_lora_to_mlp": True,
            "apply_lora_to_output": False,
            "lora_rank": 8,
            "lora_alpha": 16,
        },
        "metric_logger": {
            "_component_": "torchtune.utils.metric_logging.WandBLogger",
            "log_dir": "${output_dir}",
        },
        "optimizer": {
            "_component_": "torch.optim.AdamW",
            "lr": 3e-4,
            "weight_decay": 0.01,
        },
        "lr_scheduler": {
            "_component_": "torchtune.modules.get_cosine_schedule_with_warmup",
            "num_warmup_steps": 100,
        },
        "loss": {
            "_component_": "torch.nn.CrossEntropyLoss",
        },
        "checkpointer": {
            "_component_": "torchtune.utils.FullModelMetaCheckpointer",
            "checkpoint_dir": "${model_dir}/original",
            "checkpoint_files": [
                "consolidated.00.pth"
            ],
            "recipe_checkpoint": None,
            "output_dir": "${output_dir}",
            "model_type": "LLAMA3",
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
