import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def format_input(row: pd.Series) -> str:
    is_correct_text = "Yes" if int(row["is_correct"]) == 1 else "No"
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Is Correct Answer: {is_correct_text}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )


def tokenize_batch(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True, max_length=256)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA training for Qwen3 MAP classification")
    parser.add_argument("--train_csv", type=Path, default=Path("./data/train.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("./output"))
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--cuda", type=str, default="0")

    parser.add_argument("--use_bnb", action="store_true")
    parser.add_argument("--no_bnb", dest="use_bnb", action="store_false")
    parser.set_defaults(use_bnb=True)

    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    train_df["Misconception"] = train_df["Misconception"].fillna("NA")
    train_df["target"] = train_df["Category"] + ":" + train_df["Misconception"]

    encoder = LabelEncoder()
    train_df["label"] = encoder.fit_transform(train_df["target"])
    n_classes = len(encoder.classes_)

    is_true = train_df["Category"].str.split("_").str[0] == "True"
    correct = train_df.loc[is_true, ["QuestionId", "MC_Answer"]].copy()
    correct["count"] = correct.groupby(["QuestionId", "MC_Answer"])["MC_Answer"].transform("count")
    correct = correct.sort_values("count", ascending=False).drop_duplicates(["QuestionId"])
    correct = correct[["QuestionId", "MC_Answer"]]
    correct["is_correct"] = 1

    train_df = train_df.merge(correct, on=["QuestionId", "MC_Answer"], how="left")
    train_df["is_correct"] = train_df["is_correct"].fillna(0).astype(int)
    train_df["text"] = train_df.apply(format_input, axis=1)

    train_clean = train_df[["text", "label"]].copy()
    train_clean["label"] = train_clean["label"].astype(np.int64)
    ds_train = Dataset.from_pandas(train_clean, preserve_index=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_kwargs = {
        "num_labels": n_classes,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if args.use_bnb:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, **model_kwargs)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        modules_to_save=["score"],
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model = model.to(dtype=torch.bfloat16)

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    ds_train = ds_train.map(lambda batch: tokenize_batch(batch, tokenizer), batched=True, remove_columns=["text"])

    train_args = TrainingArguments(
        output_dir=str(args.output_dir / "training_output"),
        do_train=True,
        do_eval=False,
        save_strategy="no",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_dir=str(args.output_dir / "logs"),
        logging_steps=100,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        remove_unused_columns=False,
        bf16=True,
        fp16=False,
        report_to="none",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        dataloader_drop_last=True,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_train,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    print(f"Saved LoRA model to: {args.output_dir}")


if __name__ == "__main__":
    main()
