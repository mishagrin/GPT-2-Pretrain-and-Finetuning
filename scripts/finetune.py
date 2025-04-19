# -*- coding: utf-8 -*-
"""
Poetry Generator Fine-Tuning

Copyright: Misha Grin
"""

#
# 01. Imports
#

import os
import argparse

from transformers import GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer
from datasets import load_from_disk


if __name__ == "__main__":

    # 01. Parse path to custom tokenizer & dataset

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tokenizer', type=str, required=True, help='Path to the pretrained tokenizer.')
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to the pretrained model.')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Path to the tokenized dataset.')
    parser.add_argument('-r', '--training', type=str, required=True, help='Path to the training dir folder.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the fine-tuned output dir folder.')
    args = parser.parse_args()

    # 02. Format path

    tokenizer_path = os.path.abspath(args.tokenizer)
    model_path = os.path.abspath(args.model)
    dataset_path = os.path.abspath(args.dataset)
    training_path = os.path.abspath(args.training)
    output_path = os.path.abspath(args.output)

    # 03. Load tokenized dataset

    tokenized_datasets = load_from_disk(dataset_path)

    # 04. Init tokenizer & DataCollator

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 05. Init pretrained model

    model = GPT2LMHeadModel.from_pretrained(model_path)

    # 06. Init training arguments & trainer

    training_args = TrainingArguments(
        output_dir=training_path,
        evaluation_strategy="epoch",
        per_device_train_batch_size=12, # the lower, the better. More oppotunity for model to update weights
        per_device_eval_batch_size=12,  # but not in this case
        lr_scheduler_type="cosine",
        learning_rate=9e-5,
        gradient_accumulation_steps=64,
        weight_decay=0.01,
        warmup_steps=300,
        num_train_epochs=10,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        data_collator=data_collator,
    )

    # 07. Begin fine-tuning

    trainer.train()

    # 08. Save the model

    trainer.save_model(output_path)