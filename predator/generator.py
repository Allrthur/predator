import os
import torch
import numpy as np
import pandas as pd
import re
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    GenerationConfig
)
from .utils import LineByLineTextDataset, BlockTextDataset

def normalize_whitespace(txtseries:pd.Series):
  for txt in txtseries:
    yield re.sub(r'\s+', ' ', txt)

class Generator:
    def __init__(
        self, texts, labels, val_texts, device="cpu", model_name_or_path="distilgpt2", dtype=torch.float32
    ):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        max_length = max(len(i) for i in self.tokenizer(texts)["input_ids"])
        self.avg_length = int(
            np.mean([len(i) for i in self.tokenizer(texts)["input_ids"]])
        )
        self.std_length = int(
            np.std([len(i) for i in self.tokenizer(texts)["input_ids"]])
        )
        print("avg_length", self.avg_length, "std_length", self.std_length)

        self.config = AutoConfig.from_pretrained(model_name_or_path)
        # self.config.pad_token_id = self.tokenizer.pad_token_id
        self.config.max_length = (self.avg_length + 3 * self.std_length) # min(
        #     self.tokenizer.model_max_length,
        #     # (self.avg_length + 3 * self.std_length),
        #     float("inf"),
        # )

        max_length = self.config.max_length

        print("max_length", max_length, self.config.max_length)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, config=self.config, torch_dtype=dtype,
            low_cpu_mem_usage=True, offload_folder="offload", offload_state_dict=False
        ).to(self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        df_train = pd.DataFrame({"text": texts, "label": labels})
        max_count_label = df_train["label"].value_counts().max()
        dfs_tmp = [df_train]
        for label, group in df_train.groupby("label"):
            dfs_tmp.append(group.sample(max_count_label - len(group), replace=True))
        df_balanced = pd.concat(dfs_tmp)
        self.train_dataset = BlockTextDataset(self.tokenizer, df_balanced["text"])
        self.val_dataset = BlockTextDataset(self.tokenizer, val_texts)

    def train(self, epochs, lr=5e-5, batch_size=1):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        training_args = TrainingArguments(
            output_dir="./output-lm",
            use_cpu=(self.device != torch.device("cuda")),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            # save_steps=10,
            save_total_limit=1,
            learning_rate=lr,
            eval_strategy="epoch",
            save_strategy="epoch",
            # logging_steps=float("inf"),
            prediction_loss_only=False,
            report_to="none",
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )
        self.trainer.train()
        return self.trainer

    def evaluate(self):
        return {
            **self.trainer.evaluate(),
            **self.trainer.evaluate(self.train_dataset, metric_key_prefix="train"),
        }

    def generate(
        self,
        input_text,
        top_k=40,
        top_p=None,
        temperature=1.0,
        repetition_penalty=1.0,
        num_return_sequences=4,
    ):
        max_length_input = (
            self.config.max_length
            if self.avg_length > self.config.max_length
            else self.config.max_length - self.avg_length
        )
        input = self.tokenizer(
            input_text,
            max_length=max_length_input,
            truncation=True,
            return_tensors="pt",
        )
        # Sanity check
        self.model.to(self.device)
        input_ids = input["input_ids"].to(self.device)
        attention_mask = input["attention_mask"].to(self.device)
        # min and max length calculation
        min_length = input_ids.shape[1] + 6
        max_length_output = min(
            input_ids.shape[1] + (self.avg_length), self.config.max_length
        )
        generation_config = GenerationConfig(
            min_length=min_length,
            max_length=max_length_output,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
        )
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config
        )
        results = self.tokenizer.batch_decode(
            output[:, input_ids.size(1) :], skip_special_tokens=True
        )
        return [
            txt
            for txt in normalize_whitespace(pd.Series(results))
            if len(txt) > 3
        ]

    def save(self, path):
        os.makedirs(f"{path}/model-lm", exist_ok=True)
        self.trainer.save_model(f"{path}/model-lm")
        self.tokenizer.save_pretrained(f"{path}/model-lm")
