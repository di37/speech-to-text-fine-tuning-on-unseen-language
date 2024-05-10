import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

from datasets import DatasetDict, load_dataset, load_from_disk

class CommonVoiceDataset:
    def __init__(self, name_dataset, language):
        self.dataset = None
        self.name_dataset = name_dataset
        self.language = language
        

    def load_dataset(self):
        self.dataset = DatasetDict()
        self.dataset["train"] = load_dataset(
            self.name_dataset, self.language, split="train+validation", token=True, trust_remote_code=True
        )
        self.dataset["test"] = load_dataset(
            self.name_dataset, self.language, split="test", token=True, trust_remote_code=True
        )

    def save_to_disk(self, local_path):
        if self.dataset is not None:
            self.dataset.save_to_disk(local_path)
        else:
            print("Dataset not loaded. Please load the dataset before saving.")

    def load_from_disk(self, local_path):
        self.dataset = load_from_disk(local_path)

    def get_dataset(self):
        return self.dataset