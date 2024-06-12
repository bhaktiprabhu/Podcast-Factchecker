"""Statistical evaluation using t-tests for Claim Detection"""

import os

import numpy as np
from optimum.bettertransformer import BetterTransformer
from scipy.stats import ttest_ind
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, Trainer, TrainingArguments

from src import utils
from src.llm_model_training.training_utils import get_dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = utils.get_src_dir_path()
saved_model_dir = os.path.join(src_dir, "saved-models")


# Preprocess the data
def tokenize_function(examples):
    """Tokenizes the input text data for model training or evaluation.

    Args:
        examples (dict): A dictionary containing the input text data to be tokenized.
                         It is expected to have a key "utterance_text" with the text to be tokenized.

    Returns:
        dict: A dictionary with tokenized input data, including input_ids, attention_mask, and potentially other tokenizer outputs.
    """
    return tokenizer(examples["utterance_text"], padding="max_length", truncation=True, max_length=256)


# Compute metrics function for evaluation
def compute_metrics(p: EvalPrediction):
    """Compute accuracy, precision, recall, and F1 score for evaluation predictions.

    Args:
        p (EvalPrediction): An object containing predictions and true labels.

    Returns:
        dict: A dictionary with the computed accuracy, precision, recall, and F1 score.
    """
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# Get Test File (Episode Id:672)
dataset = get_dataset("claim-detection")
test_dataset = dataset["test"]

# Define model directories
model_directories = {
    "AlBERT": os.path.join(saved_model_dir, "albert-cd"),
    "DistilBERT": os.path.join(saved_model_dir, "distilbert-cd"),
    "DistilRoBERTa": os.path.join(saved_model_dir, "distilroberta-cd"),
    "MobileBERT": os.path.join(saved_model_dir, "mobilebert-cd"),
}

# Tokenize datasets for each model
tokenized_datasets = {}
for model_name, model_dir in model_directories.items():
    tokenizer = AutoTokenizer.from_pretrained(model_dir, do_lower_case=True)
    tokenized_datasets[model_name] = test_dataset.map(tokenize_function, batched=True)

# Load models
models = {}
for model_name, model_dir in model_directories.items():
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)
    if model_name not in ("AlBERT", "MobileBERT"):
        model = BetterTransformer.transform(model)
    models[model_name] = model

# Define training arguments
train_output_dir = os.path.join(current_dir, "training-results")
training_args = TrainingArguments(output_dir=train_output_dir, per_device_eval_batch_size=16)

# Evaluate models multiple times and collect F1-scores
num_runs = 5
model_f1_scores = {model_name: [] for model_name in models}

for run in range(num_runs):
    print(f"Run {run + 1}/{num_runs}")
    for model_name, model in models.items():
        trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics)
        tokenized_test_dataset = tokenized_datasets[model_name]
        predictions = trainer.predict(tokenized_test_dataset)
        f1_score = predictions.metrics["test_f1"]
        model_f1_scores[model_name].append(f1_score)
        print(f"{model_name} F1-score: {f1_score}")

# Perform t-test between the best model and the others
best_model_name = max(model_f1_scores, key=lambda k: np.mean(model_f1_scores[k]))
print(f"Best model: {best_model_name}")

for model_name, scores in model_f1_scores.items():
    if model_name != best_model_name:
        t_stat, p_value = ttest_ind(model_f1_scores[best_model_name], scores)
        print(f"T-test between {best_model_name} and {model_name}: t-statistic = {t_stat}, p-value = {p_value}")
