# Fine-tuning a Speech-to-Text Whisper model on unseen language - Dhivehi language

This project focuses on fine-tuning a pre-trained Whisper model for the Dhivehi language. Model has not been pre-trained on Dhivehi language at all. The base model selected is pre-trained on Sinhalese, a closely related language, to leverage linguistic similarities for better performance.

## Creating a conda environment

Its assumed that conda is already installed and nvidia gpu is available.

1. Install ffmpeg utility:

```bash
sudo apt update
sudo apt install ffmpeg
```

2. Create a new conda environment and activate it.

```bash
conda create -n speech-to-text python=3.11
conda activate speech-to-text
```

3. Install all the requirement:

```bash
pip install -r requirements.txt
```

4. Install transformers library from source.

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers/
pip install -e .
cd ..
```

## Machine Learning

- Data Ingestion and Data Preprocessing is carried out in `workflow/01_data_prep.ipynb` notebook.
- Model Training and Evaluation is carried using `workflow/02_model_training.py` script. Model is not fine-tuned in the notebook due to previous experience of it crashing while fine-tuning.
- Prediction pipeline for the fine-tuned model is carried out in `workflow/03_model_inference.ipynb` notebook.
- Evaluation and Results stored in `data/language/dv/Results_Speech_to_Text.docx`.

We can see that from `Results_Speech_to_Text.docx`, within few hours of fine-tuning and **2000 steps**, we have reached word error rate of **11.63%** of Dhivehi language. Pretty great results produced by this model as it is completely a new language that model didnt train upon during pre-training.

For more improvements on the word error rate, we can try out more larger models - `medium` and `large`.

## References

- Made use of Whisper pre-trained model - sinhalese language.
- HuggingFace Audio Course - https://huggingface.co/learn/audio-course/en/chapter5/fine-tuning
