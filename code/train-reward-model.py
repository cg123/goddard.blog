from typing import Dict, Optional

import datasets
import transformers
import trl
import typer


def tokenize_fun(tokenizer: transformers.PreTrainedTokenizerBase, max_length: int):
    def do_tok(name: str, row: Dict, suffix: str):
        tokenized = tokenizer(
            row[name],
            max_length=max_length,
            truncation=True,
        )
        return {
            f"input_ids{suffix}": tokenized["input_ids"],
            f"attention_mask{suffix}": tokenized["attention_mask"],
        }

    def f(row):
        # set up to handle three dataset formats:
        # 1. {"input_text": str, "labels": int}
        # 2. {"chosen": str, "rejected": str}
        # 3. {"prompt": str, "chosen": str, "rejected": str}

        if "prompt" in row:
            row["chosen"] = row["prompt"] + row["chosen"]
            row["rejected"] = row["prompt"] + row["rejected"]

        if "chosen" in row:
            res = {}
            res.update(do_tok("chosen", row, "_chosen"))
            res.update(do_tok("rejected", row, "_rejected"))
            return res

        return {"labels": row["labels"], **do_tok("input_text", row, "")}

    return f


def main(
    model_path: str,
    dataset: str,
    ds_config: Optional[str] = None,
    out_path: str = "reward-model-out",
    use_trl: bool = False,
    micro_batch_size: int = 8,
    eval_dataset: Optional[str] = None,
    eval_split_size: Optional[float] = None,
    seed: int = 7,
    sequence_len: int = 1024,
    lr: float = 0.0001,
    epochs: int = 1,
    regression: bool = False,
):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=1 if regression else 2
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    tokenizer.truncation_side = "left"  # we care about the most recent tokens

    if isinstance(model, transformers.GPT2ForSequenceClassification):
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    elif isinstance(model, transformers.LlamaForSequenceClassification):
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    print(tokenizer.special_tokens_map)

    ds = datasets.load_dataset(dataset, name=ds_config)
    if "train" in ds:
        ds = ds["train"]

    ds = ds.map(
        tokenize_fun(tokenizer, sequence_len),
        num_proc=16,
        remove_columns=ds.column_names,
    )

    if eval_dataset:
        ds_e = datasets.load_dataset(eval_dataset)
    elif eval_split_size:
        splits = ds.train_test_split(test_size=eval_split_size, seed=seed)
        ds = splits["train"]
        ds_e = splits["test"]
    else:
        ds_e = None

    train_kwargs = {
        "output_dir": out_path,
        "per_device_train_batch_size": micro_batch_size,
        "per_device_eval_batch_size": micro_batch_size,
        "bf16": True,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch_fused",
        "warmup_steps": 200,
        "learning_rate": lr,
        "seed": seed,
        "evaluation_strategy": "steps" if ds_e is not None else "no",
        "eval_steps": 100,
        "save_steps": 1000,
        "report_to": "wandb",
        "logging_steps": 1,
        "num_train_epochs": epochs,
        "save_total_limit": 5,
        "load_best_model_at_end": True,
        "remove_unused_columns": False,
    }
    arg_type = transformers.TrainingArguments
    if use_trl:
        arg_type = trl.RewardConfig
        train_kwargs["max_length"] = sequence_len
        train_kwargs["gradient_checkpointing"] = False
    train_args = arg_type(**train_kwargs)

    trainer_type = trl.RewardTrainer if use_trl else transformers.Trainer
    trainer = trainer_type(
        model,
        args=train_args,
        tokenizer=tokenizer,
        train_dataset=ds,
        eval_dataset=ds_e,
    )

    trainer.train()
    model.save_pretrained(out_path, safe_serialization=True)


if __name__ == "__main__":
    typer.run(main)
