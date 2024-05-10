## 02. Model Training and Evaluation

## Imports
from stt import CommonVoiceDataset, DatasetPreprocessor, Evaluator, DataCollatorSpeechSeq2SeqWithPadding, Evaluator, training_args
from utils import MODEL_NAME, SIMILAR_TARGET_LANGUAGE, TASK, MAX_INPUT_LENGTH
from utils import NAME_DATASET, TARGET_LANGUAGE, PROCESSED_DATA_PATH
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperForConditionalGeneration
from functools import partial
from transformers import Seq2SeqTrainer

# For batching the dataset while training and evaluation
preprocessor = DatasetPreprocessor(MODEL_NAME, SIMILAR_TARGET_LANGUAGE, TASK, MAX_INPUT_LENGTH)
normalizer = BasicTextNormalizer()

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=preprocessor.processor)
evaluator = Evaluator(processor=preprocessor.processor, normalizer=normalizer)

# Loading preprocessed dataset
print("Loading preprocessed dataset ...")
dataset = CommonVoiceDataset(name_dataset=NAME_DATASET, language=TARGET_LANGUAGE)
dataset.load_from_disk(PROCESSED_DATA_PATH)
dataset = dataset.get_dataset()
print("Dataset loaded!")

# Loading Pre-Trained Checkpoint
print("Loading pre-trained model ...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# disable cache during training since it's incompatible with gradient checkpointing
model.config.use_cache = False

# set language and task for generation and re-enable cache
model.generate = partial(
    model.generate, language=SIMILAR_TARGET_LANGUAGE, task=TASK, use_cache=True
)
print("Model loaded!")

# Training
print("Training commenced ...")
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=evaluator.compute_metrics,
    tokenizer=preprocessor.processor,
)

trainer.train()
print("Training completed!")