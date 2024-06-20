import json
import os
from typing import List
import sys
import datasets
VALUE = 100000

considered_datasets = {"SODA-inst (T)": VALUE, "Xlel_wd-inst (T)": VALUE, "HotpotQA (T)": VALUE, "CNN-Daily-Mail (T)": VALUE,
                       "NQ-Open (T)": VALUE, "Mintaka-inst (T)": VALUE, "MLQA-en (T)": VALUE, "PIQA (T)": VALUE, 
                       "Flan-CoT-submix (T)": VALUE, "Flan-Coqa (T)": VALUE, "Flan-unified-QA (T)": VALUE}

so_far = {}



_URL = "https://huggingface.co/datasets/CohereForAI/aya_collection_language_split/resolve/main/romanian/train-00000-of-00001.parquet"
_URLS = {
    "train": [
        _URL + "train-00000-of-00001.parquet",
    ],
    # "test": [
    #     _URL + "test-00000-of-00001.parquet",
    # ],
    # "validation": [
        # _URL + "validation-00000-of-00001.parquet",
    # ],
}

class Aya(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    def _info(self):
        features = datasets.Features(
            {"prompt": datasets.Value("string"),
             "response": datasets.Value("string")}
        )
        return datasets.DatasetInfo(features=features)

    def _split_generators(self, dl_manager):
        file_path = dl_manager.download(_URL)
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": file_path})]

    def _generate_examples(self, filepath: str):
        global considered_datasets
        dataset = datasets.load_dataset("parquet", data_files=filepath)
        entry_id = -1
        added_ds = []
        ds = set()
        for data_entry in dataset["train"]:
            ds.add(data_entry["dataset_name"])
            if data_entry["dataset_name"] in considered_datasets:
                if data_entry["dataset_name"] == "SODA-inst (T)":
                    data_entry["inputs"] = data_entry["inputs"].replace("  ", " ")
                    if not data_entry["inputs"].endswith("."):
                        data_entry["inputs"] = data_entry["inputs"] + "."

                elif data_entry["dataset_name"] == "Xlel_wd-inst (T)":
                    if data_entry["inputs"] == "Completați următoarea frază:":
                        continue
                    if data_entry["targets"].startswith("- Da, domnule."):
                        continue
                    data_entry["targets"] = " " + data_entry["targets"]

                elif data_entry["dataset_name"] == "HotpotQA (T)":
                    if data_entry["inputs"] == "Răspundeţi la această întrebare complicată:":
                        continue
                    if data_entry["inputs"] == "Formulează un răspuns la această întrebare elaborată:":
                        continue
                    
                    data_entry["inputs"] = data_entry["inputs"].replace(" elaborată", "").replace(" complicată", "")
                    data_entry["inputs"] = data_entry["inputs"].replace("Răspunde la această întrebare: ", "").replace("Formulează un răspuns la această întrebare: ", "").replace("Răspundeţi la această întrebare: ", "").replace("Formulaţi un răspuns la această întrebare: ", "").replace("Răspundeţi la întrebarea: ", "")
                    if data_entry["targets"] == "Da, domnule.":                                                                                                                   
                        data_entry["targets"] = "Da."
                    if data_entry["targets"] == "Nu, nu.":
                        data_entry["targets"] = "Nu."
                    if not data_entry["targets"].endswith("."):
                        data_entry["targets"] = data_entry["targets"] + "."

                    # data_entry["inputs"] = "Întrebare: " + data_entry["inputs"] + "\nRăspuns:"
                    # data_entry["targets"] = "Răspuns: " + data_entry["targets"]
                
                elif data_entry["dataset_name"] == "NQ-Open (T)":
                    question = data_entry["inputs"].split(" Răspuns:")[0]
                    if not question.endswith("?"):
                        question = question + "?"
                    data_entry["inputs"] = question + "\nRăspuns:"

                elif data_entry["dataset_name"] == "Mintaka-inst (T)":
                    # do nothing
                    pass
                
                elif data_entry["dataset_name"] == "MLQA-en (T)":
                    pass
                
                elif data_entry["dataset_name"] == "PIQA (T)":
                    if data_entry["inputs"].startswith("Termină următoarea propoziţie cu cea mai bună alegere:  Alegeri:"):
                        continue
                    data_entry["inputs"] = data_entry["inputs"].replace("  Alegeri:", "\nAlegeri:").replace(" - ", "\n- ").replace("  Răspuns:", "\nRăspuns:")

                elif data_entry["dataset_name"] == "Flan-CoT-submix (T)":
                    data_entry["inputs"] = data_entry["inputs"].replace("  Opțiuni:", "\nOpțiuni:").replace(" - ", "\n- ").replace("  Răspuns:", "\nRăspuns:")

                elif data_entry["dataset_name"] == "Flan-unified-QA (T)":
                    pass


                if data_entry["dataset_name"] in so_far:
                    if so_far[data_entry["dataset_name"]] != 0:
                        so_far[data_entry["dataset_name"]] += 1
                        if so_far[data_entry["dataset_name"]] == considered_datasets[data_entry["dataset_name"]]:
                            del considered_datasets[data_entry["dataset_name"]]
                            so_far[data_entry["dataset_name"]] = 0
                else:
                    so_far[data_entry["dataset_name"]] = 1
                
                entry_id += 1
                # print({"prompt": data_entry["inputs"], "response": data_entry["targets"]})
                added_ds.append(data_entry["dataset_name"])
                yield entry_id, {
                    "prompt": data_entry["inputs"],
                    "response": data_entry["targets"]
                }

        # from collections import Counter
        # print("Added datasets:", Counter(added_ds))
        # print("All datasets:", ds)
        # sys.exit()
        