import json
import random
import os
from os.path import join


class SFTFormatter:

    def __init__(self, data_dir,prompt_template,raw_path):
        

        self.data_dir = data_dir
        self.prompt_template = prompt_template

        self.raw_path = raw_path

        self.output_dir = join(self.data_dir,
            "datasets",
            "llamafactory-finetune-data"
        )

        os.makedirs(self.output_dir, exist_ok=True)

        self.dataset = []

    # -----------------------------

    def load_data(self):
        """Load jsonl dataset"""

        data = []

        with open(self.raw_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                data.append(json.loads(line.strip()))

        return data

    # -----------------------------

    def format(self, records):
        """Build fine-tuning format"""

        formatted = []

        for rec in records:

            system_message = self.prompt_template

            formatted.append({
                "system": system_message,

                "instruction": "\n".join([
                    "# Story:",
                    rec["story"],

                    "# Task:",
                    rec["task"],

                    "# Output Scheme:",
                    rec["output_scheme"],

                    "",
                    "# Output JSON:",
                    "```json"
                ]),

                "input": "",

                "output": "\n".join([
                    "```json",
                    json.dumps(
                        rec["response"],
                        ensure_ascii=False,
                        default=str
                    ),
                    "```"
                ]),

                "history": []
            })

        return formatted


    def split(self, dataset):
        """Train / Validation split"""

        random.Random(self.config.get("seed", 101)).shuffle(dataset)

        train_size = self.config.get("train_size", 2700)

        train_ds = dataset[:train_size]
        val_ds = dataset[train_size:]

        return train_ds, val_ds



    def save(self, train_ds, val_ds):
        """Save datasets"""

        train_path = join(self.output_dir, "train.json")
        val_path = join(self.output_dir, "val.json")

        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_ds, f, ensure_ascii=False, default=str, indent=2)

        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(val_ds, f, ensure_ascii=False, default=str, indent=2)


    def run(self):
        records = self.load_data()
        dataset = self.format(records)

        train_ds, val_ds = self.split(dataset)

        self.save(train_ds, val_ds)

        print("SFT Dataset formatting completed.")