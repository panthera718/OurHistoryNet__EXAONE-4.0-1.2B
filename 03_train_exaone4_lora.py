import os
import re
from typing import List

import torch
from datasets import load_dataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    __version__ as transformers_version,
)
from peft import LoraConfig, get_peft_model

# ==========================
# Paths / knobs
# ==========================
# Option A) Load directly from Hugging Face Hub:
BASE_MODEL_PATH = "./EXAONE-4.0-1.2B"
# Option B) If you downloaded the model locally, set BASE_MODEL_PATH to that folder.

DATA_PATH   = "history_sft_train.jsonl"
OUTPUT_DIR  = "./EXAONE4-history-lora"
MAX_SEQ_LEN = 2048

# Typical LLaMA-style module names (EXAONE 4.0 uses an exaone4 architecture in Transformers)
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

def _check_transformers_version():
    # EXAONE 4.0 is officially supported in transformers >= 4.54.0
    if version.parse(transformers_version) < version.parse("4.54.0"):
        raise RuntimeError(
            f"Transformers >= 4.54.0 is required for EXAONE 4.0, but you have {transformers_version}. "
            "Please upgrade: pip install -U 'transformers>=4.54.0'"
        )

def _pick_dtype():
    # The model card recommends bfloat16; fall back to fp16 if bf16 isn't available.
    if torch.cuda.is_available():
        try:
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32

def _resolve_target_modules(model) -> List[str]:
    """
    Try to verify that DEFAULT_TARGET_MODULES exist in the model.
    If not, attempt a best-effort fallback by searching common projection layer suffixes.
    """
    module_names = set()
    for name, _ in model.named_modules():
        module_names.add(name.split(".")[-1])

    if all(m in module_names for m in DEFAULT_TARGET_MODULES):
        return DEFAULT_TARGET_MODULES

    # Best-effort fallback (keeps training from silently doing nothing)
    candidates = []
    for m in DEFAULT_TARGET_MODULES:
        if m in module_names:
            candidates.append(m)

    if candidates:
        print("[WARN] Some target modules were not found. Using the subset that exists:", candidates)
        return candidates

    # Last fallback for GPT-2 style names (just in case)
    gpt2_like = ["c_attn", "c_proj", "c_fc"]
    if any(m in module_names for m in gpt2_like):
        print("[WARN] Falling back to GPT-2 style target modules:", gpt2_like)
        return gpt2_like

    raise RuntimeError(
        "Could not resolve LoRA target modules. "
        "Please inspect `print([n for n,_ in model.named_modules()])` and update target_modules."
    )

def print_trainable_parameters(model):
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"Trainable params: {trainable} / {total} ({100 * trainable / total:.2f}%)")

def main():
    _check_transformers_version()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    # EXAONE provides PAD/BOS/EOS in tokenizer config, but keep a safety guard.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = _pick_dtype()
    print(f"[INFO] Loading base model with dtype={dtype} ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=dtype,
        device_map="auto",
    )

    # For memory saving
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    target_modules = _resolve_target_modules(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    print(f"[INFO] Loading dataset from {DATA_PATH} ...")
    raw_datasets = load_dataset("json", data_files={"train": DATA_PATH})

    def format_example(example):
        instruction = example.get("instruction", "").strip()
        context     = example.get("input", "").strip()
        answer      = example.get("output", "").strip()

        if context:
            prompt = (
                "다음은 국사편찬위원회 우리역사넷(신편 한국사)에 기반한 질의응답이다.\n"
                "주어진 참고 자료를 우선적으로 활용하여, 우리역사넷의 서술에서 벗어나지 않도록 신중하게 답변하라.\n\n"
                f"질문: {instruction}\n\n"
                f"참고 자료: {context}\n\n"
                "답변: "
            )
        else:
            prompt = (
                "다음은 국사편찬위원회 우리역사넷(신편 한국사)에 기반한 질의응답이다.\n"
                "우리역사넷의 일반적인 서술 방식과 내용을 따라, 신중하게 답변하라.\n\n"
                f"질문: {instruction}\n\n"
                "답변: "
            )

        full_text = prompt + answer + tokenizer.eos_token
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("[INFO] Tokenizing dataset ...")
    tokenized_train = raw_datasets["train"].map(
        format_example,
        remove_columns=raw_datasets["train"].column_names,
        batched=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Use bf16 if possible; else fp16 (keeps your original intent)
    use_bf16 = (dtype == torch.bfloat16)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-4,
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator,
    )

    print("[INFO] Start training (LoRA) ...")
    trainer.train()

    print("[INFO] Saving LoRA adapter and tokenizer ...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[DONE] LoRA fine-tuning finished. Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
