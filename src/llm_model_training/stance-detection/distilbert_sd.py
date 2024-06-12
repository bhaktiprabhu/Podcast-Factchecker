"""Training of DistilBERT Transformer model with Podcast Stance Detection Data"""

import os

import wandb
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, TrainingArguments

from src import utils
from src.llm_model_training.training_utils import CustomTrainer, compute_metrics, get_class_weights_train, get_dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = utils.get_src_dir_path()
saved_model_dir = os.path.join(src_dir, "saved-models")

# Class distributions in training set
num_class_0_train = 141
num_class_1_train = 343
class_weights_train = get_class_weights_train(num_class_0_train, num_class_1_train)
print("Class Weights Normalized: ", class_weights_train)


# Define label mapping
label_to_id = {"SUPPORTS": 1, "REFUTES": 0}


# Preprocess the data
def tokenize_function(examples):
    """Tokenizes the input text data for model training or evaluation.

    Args:
        examples (dict): A dictionary containing the input text data to be tokenized.

    Returns:
        dict: A dictionary with tokenized input data, including input_ids, attention_mask, and potentially other tokenizer outputs.
    """
    combined_texts = [
        "[Claim]: " + claim + " [Evidence]: " + evidence
        for claim, evidence in zip(examples["check_worthy_claim"], examples["evidence_snippet"])
    ]
    return tokenizer(combined_texts, padding="max_length", truncation=True, max_length=512)


def map_labels(examples):
    """Function to map string labels to integers"""

    examples["label"] = [label_to_id[label] for label in examples["label"]]
    return examples


################################################################################################

dataset = get_dataset("stance-detection")

# Apply label mapping
dataset = dataset.map(map_labels, batched=True)

train_dataset = dataset["train"]
validation_dataset = dataset["validation"]

print("---------------Train Dataset------------------")
print(train_dataset)
print(train_dataset.shape)
print(train_dataset.size_in_bytes)

print("------------Validation Dataset---------------")
print(validation_dataset)
print(validation_dataset.shape)
print(validation_dataset.size_in_bytes)

wandb.init(project="podcast-stance-detection", dir=current_dir)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=True)
auto_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

print("------------------------------------------------------------------------------------------")
print(
    f"Pre-Trained Model 'distilbert-base-uncased' has been Loaded: \
    It has been trained on {sum(p.numel() for p in auto_model.parameters() if p.requires_grad)} parameters."
)
print("------------------------------------------------------------------------------------------")

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = validation_dataset.map(tokenize_function, batched=True)

# Remove the original text columns after tokenization
tokenized_train_dataset = tokenized_train_dataset.remove_columns(["check_worthy_claim", "evidence_snippet"])
tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(["check_worthy_claim", "evidence_snippet"])

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
trainer = CustomTrainer(
    class_weights=class_weights_train,
    model=auto_model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    compute_metrics=compute_metrics,  # Pass the compute_metrics function
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],  # Used while Training to stop if eval loss increases
)

trainer.train()

model_path = os.path.join(saved_model_dir, "distilbert-podcast-stance-detection")
auto_model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)


# Finish the wandb run when training is done
wandb.finish()
