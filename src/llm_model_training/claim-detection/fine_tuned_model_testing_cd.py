"""Testing Fine-Tuned Models for Claim Detection"""

import os

import numpy as np
from optimum.bettertransformer import BetterTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
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

    # preds = p.predictions.argmax(-1)
    # probs = torch.nn.functional.softmax(torch.tensor(p.predictions), dim=-1)
    # adjusted_preds = (probs[:, 1] > 0.2).int()  # Adjust threshold here

    conf_matrix = confusion_matrix(labels, preds)
    print(conf_matrix)

    report = classification_report(labels, preds, target_names=["not_cw_claim", "is_cw_claim"])

    # Micro average metrics
    micro_precision, micro_recall, micro_f1_score, _ = precision_recall_fscore_support(labels, preds, average="micro")

    # Append Micro Avg Metrics
    report += f"   micro avg       {micro_precision:.2f}      {micro_recall:.2f}      {micro_f1_score:.2f}       {len(labels)}"

    print(report)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


################################################################################################

# Get Test File (Episode Id:672)
dataset = get_dataset("claim-detection")
test_dataset = dataset["test"]

print("---------------Test Dataset------------------")
print(test_dataset)
print(test_dataset.shape)
print(test_dataset.size_in_bytes)


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


# Get Predictions and Evaluations for Testing the Model
for model_name, model in models.items():
    trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics)
    tokenized_test_dataset = tokenized_datasets[model_name]
    print("-------------------------------------------------------------------------------------------")
    print(f"**** {model_name} Podcast Claim Detection Model Evaluation for Test Data ****")
    predictions = trainer.predict(tokenized_test_dataset)
    print(predictions.metrics)
