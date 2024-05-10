import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

from transformers import pipeline
from utils import TASK, SIMILAR_TARGET_LANGUAGE

class SpeechTranscriber:
    def __init__(self, model_id):
        self.model_id = model_id
        self.pipe = self.load_model(self.model_id)
    
    def load_model(self, model_id):
        self.pipe = pipeline("automatic-speech-recognition", model=model_id)

    def transcribe_speech(self, filepath):
        output = self.pipe(
            filepath,
            max_new_tokens=256,
            generate_kwargs={
                "task": TASK,
                "language": SIMILAR_TARGET_LANGUAGE,
            },  # update with the language you've fine-tuned on
            chunk_length_s=30,
            batch_size=8,
        )
        return output["text"]