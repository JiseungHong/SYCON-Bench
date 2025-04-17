import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_dataset_from_file(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return Dataset.from_dict({"text": lines})

def tokenize_function(example, tokenizer):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

def finetune_with_lora(dataset_path, output_dir, epochs=1):
    print(f"Fine-tuning on: {dataset_path}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load and tokenize
    dataset = load_dataset_from_file(dataset_path)
    tokenized = dataset.map(lambda ex: tokenize_function(ex, tokenizer), batched=True)

    # Load model with LoRA
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        load_in_4bit=True
    )
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=epochs,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    print(f"Saved LoRA model to: {output_dir}")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    finetune_with_lora(
        dataset_path="analysis/data/positive_trajectory.txt",
        output_dir="analysis/model/pos_lora",
        epochs=1
    )

    finetune_with_lora(
        dataset_path="analysis/data/negative_trajectory.txt",
        output_dir="analysis/model/neg_lora",
        epochs=1
    )
