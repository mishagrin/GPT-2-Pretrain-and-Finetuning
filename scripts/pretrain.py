# -*- coding: utf-8 -*-
"""
Poetry Generator Pretraining

Copyright: Misha Grin
"""

#
# 01. Imports
#

import os
import argparse

from transformers import GPT2LMHeadModel, AutoConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer
from datasets import load_from_disk


if __name__ == "__main__":
    
    # 01. Parse path to custom tokenizer & dataset

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tokenizer', type=str, required=True, help='Path to the pretrained tokenizer.')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Path to the tokenized dataset.')
    parser.add_argument('-r', '--training', type=str, required=True, help='Path to the training dir folder.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the output dir folder.')
    args = parser.parse_args()

    # 02. Format path

    tokenizer_path = os.path.abspath(args.tokenizer)
    dataset_path = os.path.abspath(args.dataset)
    training_path = os.path.abspath(args.training)
    output_path = os.path.abspath(args.output)

    # 03. Load tokenized dataset

    tokenized_datasets = load_from_disk(dataset_path)

    # 04. Load pretrained tokenizer 

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 05. Create GPT2 configuration

    context_length = 512

    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # 06. Init GPT2 model

    model = GPT2LMHeadModel(config)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

    # 07. Data Collator

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 08. Check data collator example

    out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
    for key in out:
        print(f"{key} shape: {out[key].shape}")

    # 09. Training Arguments

    args = TrainingArguments(
    output_dir=training_path,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=64,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=True,
    push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )

    # 10. Begin pretraining

    trainer.train()

    # 10. Save result

    trainer.save_model(output_path)