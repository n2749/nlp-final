import os
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "results"
SEED = 42
NUM_EPOCHS = 3
BATCH_SIZE = 8
MAX_LENGTH = 128
set_seed(SEED)

raw_datasets = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_fn(examples):
    return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

tokenized_datasets = raw_datasets.map(
    preprocess_fn,
    batched=True,
    remove_columns=["sentence", "idx"]
)

tokenized_datasets.set_format("torch")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    seed=SEED
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

eval_results = trainer.evaluate()
print("=== Evaluation Results ===")
for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")

history = trainer.state.log_history

train_epochs = [e["epoch"] for e in history if "loss" in e and e.get("epoch") is not None]
train_losses = [e["loss"] for e in history if "loss" in e and e.get("epoch") is not None]

eval_epochs = [e["epoch"] for e in history if e.get("eval_accuracy") is not None]
eval_accuracy = [e["eval_accuracy"] for e in history if e.get("eval_accuracy") is not None]

plt.figure(figsize=(8, 5))
plt.plot(train_epochs, train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Learning curve (loss)")
plt.grid(True)
plt.show()
plt.figure(figsize=(8, 5))
plt.plot(eval_epochs, eval_accuracy, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Accuracy on validation")
plt.grid(True)
plt.show()