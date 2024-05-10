import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

import evaluate
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import Seq2SeqTrainingArguments
from utils import MODEL_PATH, PER_DEVICE_TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE, LR_SCHEDULER_TYPE, WARMUP_STEPS, MAX_STEPS, GRADIENT_CHECKPOINTING, FP16, FP16_FULL_EVAL, EVALUATION_STRATEGY, PER_DEVICE_EVAL_BATCH_SIZE, PREDICT_WITH_GENERATE, GENERATION_MAX_LENGTH, SAVE_STEPS, EVAL_STEPS, LOGGING_STEPS, LOAD_BEST_MODEL_AT_END, METRIC_FOR_BEST_MODEL, GREATER_IS_BETTER, PUSH_TO_HUB, LOGGING_DIR

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


class Evaluator:
    def __init__(self, processor, normalizer):
        self.processor = processor
        self.normalizer = normalizer
        self.metric = evaluate.load("wer")

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)

        wer_ortho = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        pred_str_norm = [self.normalizer(pred) for pred in pred_str]
        label_str_norm = [self.normalizer(label) for label in label_str]

        pred_str_norm = [
            pred_str_norm[i]
            for i in range(len(pred_str_norm))
            if len(label_str_norm[i]) > 0
        ]
        label_str_norm = [
            label_str_norm[i]
            for i in range(len(label_str_norm))
            if len(label_str_norm[i]) > 0
        ]

        wer = 100 * self.metric.compute(predictions=pred_str_norm, references=label_str_norm)

        return {"wer_ortho": wer_ortho, "wer": wer}
    

training_args = Seq2SeqTrainingArguments(
    output_dir=MODEL_PATH,  # name on the HF Hub
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # increase by 2x for every 2x decrease in batch size
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,  # increase to 4000 if you have your own GPU or a Colab paid plan
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    fp16=FP16,
    fp16_full_eval=FP16_FULL_EVAL,
    evaluation_strategy=EVALUATION_STRATEGY,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    predict_with_generate=PREDICT_WITH_GENERATE,
    generation_max_length=GENERATION_MAX_LENGTH,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    logging_steps=LOGGING_STEPS,
    logging_dir=LOGGING_DIR,
    save_total_limit=2,
    load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
    metric_for_best_model=METRIC_FOR_BEST_MODEL,
    greater_is_better=GREATER_IS_BETTER,
    push_to_hub=PUSH_TO_HUB,
)