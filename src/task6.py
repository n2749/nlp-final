from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt


def get_conll_dataset():
    dataset = load_dataset("conll2003")
    return dataset


def tokenize_and_align_labels(dataset_split, tokenizer, label_all_tokens=True):
    label_list = dataset_split.features["ner_tags"].feature.names

    def tokenize(example):
        tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != previous_word_idx:
                labels.append(example["ner_tags"][word_idx])
            else:
                labels.append(example["ner_tags"][word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_split = dataset_split.map(tokenize, batched=False)
    return tokenized_split, label_list


accuracy_per_epoch = []

def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)

    true_labels = [l for label in labels for l in label if l != -100]
    true_preds = [p for prediction, label in zip(predictions, labels) for p, l in zip(prediction, label) if l != -100]

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_preds, average="weighted")
    accuracy = np.mean(np.array(true_preds) == np.array(true_labels))
    accuracy_per_epoch.append(accuracy)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def plot_accuracy(accuracies):
    plt.figure()
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o')
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_per_epoch.png")


def main():
    model_checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=9)

    dataset = get_conll_dataset()
    tokenized_train, label_list = tokenize_and_align_labels(dataset["train"], tokenizer)
    tokenized_val, _ = tokenize_and_align_labels(dataset["validation"], tokenizer)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir="./bert-conll2003-ner",
        evaluation_strategy="epoch",
        logging_dir="./logs",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    plot_accuracy(accuracy_per_epoch)
    metrics = trainer.evaluate()
    print("\nFinal Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()

