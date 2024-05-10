import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

NAME_DATASET = "mozilla-foundation/common_voice_16_1"
TARGET_LANGUAGE = "dv"
SIMILAR_TARGET_LANGUAGE = "sinhalese"
RAW_DATA_PATH = "data/language/dv/raw"
PROCESSED_DATA_PATH = "data/language/dv/processed"
MODEL_NAME = "openai/whisper-small"
TASK = "transcribe"
MODEL_PATH = "models/whisper-small-dv"
NUM_PROC = 1
MAX_INPUT_LENGTH = 30.0

# Hyperparameters
PER_DEVICE_TRAIN_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2  # increase by 2x for every 2x decrease in batch size
LEARNING_RATE = 1e-5
LR_SCHEDULER_TYPE = "constant_with_warmup"
WARMUP_STEPS = 50
MAX_STEPS = 2000  # increase to 4000 if you have your own GPU or a Colab paid plan
GRADIENT_CHECKPOINTING = True
FP16 = True
FP16_FULL_EVAL = True
EVALUATION_STRATEGY = "steps"
PER_DEVICE_EVAL_BATCH_SIZE = 16
PREDICT_WITH_GENERATE = True
GENERATION_MAX_LENGTH = 225
SAVE_STEPS = 500
EVAL_STEPS = 500
LOGGING_STEPS = 50
LOGGING_DIR = "logs"
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "wer"
GREATER_IS_BETTER = False
PUSH_TO_HUB = False