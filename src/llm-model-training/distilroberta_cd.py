"""Training of Distilbert Transformer model with Podcast Data"""

import os

import numpy as np
import torch
import wandb
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from src import utils

current_dir = os.path.dirname(__file__)


# Class distributions in training set
num_class_0_train = 1220  # number of samples for class 0 (False/Not-CW) in training set
num_class_1_train = 199  # number of samples for class 1 (True/CW) in training set
total_samples_train = num_class_0_train + num_class_1_train

# Calculate the Class Weights Using Inverse Frequency
weight_for_class_0_train = total_samples_train / num_class_0_train
weight_for_class_1_train = total_samples_train / num_class_1_train

print("Class Weights: ", [weight_for_class_0_train, weight_for_class_1_train])

# Normalize Weights
total_weight = weight_for_class_0_train + weight_for_class_1_train
weight_class_0_normalized = weight_for_class_0_train / total_weight
weight_class_1_normalized = weight_for_class_1_train / total_weight

print("Class Weights Normalized: ", [weight_class_0_normalized, weight_class_1_normalized])


class_weights_train = torch.tensor([weight_class_0_normalized, weight_class_1_normalized])


class FocalLoss(torch.nn.Module):
    """Focal Loss for addressing class imbalance in classification tasks."""

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """Computes the focal loss between `inputs` and `targets`."""
        ce_loss = torch.nn.CrossEntropyLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-ce_loss)  # pylint: disable=invalid-unary-operand-type
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class FocalTrainer(Trainer):
    """Custom Trainer class that uses Focal Loss for training."""

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Class frequencies
        alpha = class_weights_train.to(logits.device)

        # Custom Focal Loss function
        loss_fct = FocalLoss(alpha=alpha, gamma=2.5)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


class CustomTrainer(Trainer):
    """Custom Trainer class to compute custom loss"""

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_train.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


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

    # probs = torch.nn.functional.softmax(torch.tensor(p.predictions), dim=-1)
    # preds = (probs[:, 1] > 0.3).int()  # Adjust threshold here
    # labels = p.label_ids

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


################################################################################################

# Get Train, Validation and Test Files
eps = utils.list_all_episodes()
# eps = [219, 555, 562, 563, 564, 565, 566, 630, 672]

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.abspath(current_dir))

train_files, validation_files, test_file = [], [], []

data_source_path = os.path.join(parent_dir, "output", "crowd-work", "claim-detection-ground-truth")

for i, ep_id in enumerate(eps):
    podcast_dir = utils.get_podcast_dir_path(data_source_path, ep_id)
    filename = f"cd_ground_truth_ep_{ep_id}.csv"
    file_path = os.path.join(podcast_dir, filename)
    if i < 7:
        train_files.append(file_path)
    elif i < 9:
        validation_files.append(file_path)
    else:
        test_file.append(file_path)

# print(train_files)
# print(validation_files)
# print(test_file)

# Load the dataset
dataset = load_dataset("csv", data_files={"train": train_files, "validation": validation_files, "test": test_file})
print("---------------Dataset Loaded------------------")
print(dataset)

train_dataset = dataset["train"].remove_columns(["utterance_id"]).rename_column("is_check_worthy_claim", "label")
validation_dataset = (
    dataset["validation"].remove_columns(["utterance_id"]).rename_column("is_check_worthy_claim", "label")
)
test_dataset = dataset["test"].remove_columns(["utterance_id"]).rename_column("is_check_worthy_claim", "label")

print("---------------Train Dataset------------------")
print(train_dataset)
print(train_dataset.shape)
print(train_dataset.size_in_bytes)

print("------------Validation Dataset---------------")
print(validation_dataset)
print(validation_dataset.shape)
print(validation_dataset.size_in_bytes)

wandb.init(project="podcast-claim-detection", dir=current_dir)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base", do_lower_case=True)
auto_model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2)

print("------------------------------------------------------------------------------------------")
print(
    f"Pre-Trained Model 'distilroberta-base' has been Loaded: \
    It has been trained on {sum(p.numel() for p in auto_model.parameters() if p.requires_grad)} parameters."
)
print("------------------------------------------------------------------------------------------")

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = validation_dataset.map(tokenize_function, batched=True)

# print(tokenized_train_dataset[50])
# print(tokenized_eval_dataset[50])

train_ouptut_dir = os.path.join(current_dir, "training-results")

# Define training arguments
training_args = TrainingArguments(
    output_dir=train_ouptut_dir,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=0,
    weight_decay=0.01,
    logging_strategy="epoch",  # Log at the end of every epoch
    evaluation_strategy="epoch",  # Evaluate at the end of every epoch
    save_strategy="epoch",  # Save at the end of every epoch
    report_to="wandb",  # Enable logging to wandb
    do_train=True,
    do_eval=True,
    load_best_model_at_end=True,  # Used for Early Stopping while training
    metric_for_best_model="loss",  # Default is Evaluation Loss ("loss"), Can be any eval metric like eval_f1
)


# Create a Trainer
trainer = FocalTrainer(
    model=auto_model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    compute_metrics=compute_metrics,  # Pass the compute_metrics function
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],  # Used while Training to stop if eval loss increases
)


trainer.train()

model_path = os.path.join(current_dir, "saved-models", "distilroberta-focal")
auto_model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# trainer.evaluate()
# predictions = trainer.predict(test_dataset)

# Finish the wandb run when training is done
wandb.finish()
