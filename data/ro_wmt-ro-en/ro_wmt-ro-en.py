import json
import os
from typing import List
import sys
import datasets

description = "Tradu următorul text din română în engleză: "

class WMTRoEN(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    def _info(self):
        features = datasets.Features(
            {"prompt": datasets.Value("string"),
             "response": datasets.Value("string")}
        )
        return datasets.DatasetInfo(features=features)

    def _split_generators(self, dl_manager):
        file_path = dl_manager.download("https://huggingface.co/datasets/wmt/wmt16/resolve/main/ro-en/train-00000-of-00001.parquet")
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": file_path})]

    def _generate_examples(self, filepath: str):
        dataset = datasets.load_dataset("parquet", data_files=filepath)
        entry_id = -1
        for data_entry in dataset["train"]:
            entry_id += 1
            yield entry_id, {
                "prompt": "{0}{1}\n".format(description, data_entry["translation"]["ro"]),
                "response": data_entry["translation"]["en"]
            }

        