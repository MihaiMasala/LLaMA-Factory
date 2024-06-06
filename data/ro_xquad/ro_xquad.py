import json
import os
from typing import List
import sys
import datasets


class XQUAD(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    def _info(self):
        features = datasets.Features(
            {"prompt": datasets.Value("string"),
             "response": datasets.Value("string")}
        )
        return datasets.DatasetInfo(features=features)

    def _split_generators(self, dl_manager):
        file_path = dl_manager.download("https://huggingface.co/datasets/rajpurkar/squad/resolve/main/plain_text/train-00000-of-00001.parquet")
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": file_path})]

    def _generate_examples(self, filepath: str):
        dataset = datasets.load_dataset("parquet", data_files=filepath)
        entry_id = -1
        for data_entry in dataset["train"]:
            entry_id += 1
            yield entry_id, {
                "prompt": "{0}\nÎntrebare: {1}\nRăspuns:".format(data_entry["context"], data_entry["question"]),
                "response": data_entry["answers"]["text"][0]
            }       