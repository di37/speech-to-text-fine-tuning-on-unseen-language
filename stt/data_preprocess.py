import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

from transformers import WhisperProcessor
from datasets import Audio

class DatasetPreprocessor:
    def __init__(self, model_name, similar_target_language, task, max_input_length):
        self.model_name = model_name
        self.similar_target_language = similar_target_language
        self.task = task
        self.max_input_length = max_input_length
        self.processor = WhisperProcessor.from_pretrained(
            self.model_name, language=self.similar_target_language, task=self.task
        )
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

    def prepare_dataset(self, dataset, num_proc):
        dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        dataset = dataset.map(
            self.process_example,
            remove_columns=dataset.column_names["train"],
            num_proc=num_proc,
        )
        return dataset

    def process_example(self, example):
        audio = example["audio"]
        example = self.processor(
            audio=audio["array"],
            sampling_rate=audio["sampling_rate"],
            text=example["sentence"],
        )
        example["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        return example

    def filter_dataset(self, dataset):
        dataset["train"] = dataset["train"].filter(
            self.is_audio_in_length_range,
            input_columns=["input_length"],
        )
        return dataset

    def is_audio_in_length_range(self, length):
        return length < self.max_input_length