import json
import os
from typing import List
import sys
import datasets

description = "Generează un număr între 0 și 1 care descrie similaritatea semantică dintre următoarele două propoziții:"

class STS(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    def _info(self):
        features = datasets.Features(
            {"prompt": datasets.Value("string"),
             "response": datasets.Value("string")}
        )
        return datasets.DatasetInfo(features=features)

    def _split_generators(self, dl_manager):
        file_path = "data/ro_sts_train.json"
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": file_path})]

    def _generate_examples(self, filepath: str):
        dataset = datasets.load_dataset("json", data_files=filepath)
        entry_id = -1
        for data_entry in dataset["train"]:
            print(data_entry)
            # Propoziție 1: {{sentence1}}\nPropoziție 2: {{sentence2}}\nScor de similaritate semantică:
            # sys.exit()
            entry_id += 1
            yield entry_id, {
                "prompt": "{0}\nPropoziție 1: {1}\nPropoziție 2: {2}\nScor de similaritate semantică:".format(description, data_entry["sentence1"], data_entry["sentence2"]),
                "response": data_entry["score"]
            }       