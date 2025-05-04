from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import evaluate


def get_conll_dataset():
    dataset = load_dataset("conll2003")
    return dataset


def tokenize_and_align_labels(dataset, tokenizer, label_all_tokens=True):
    label_list = dataset["train"].features["ner_tags"].feature.names

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

    return dataset.map(tokenize, batched=True), label_list


def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)

    true_labels = [l for label in labels for l in label if l != -100]
    true_preds = [p for prediction, label in zip(predictions, labels) for p, l in zip(prediction, label) if l != -100]

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_preds, average="weighted")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    model_checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=9)  # 9 = number of CoNLL-2003 labels

    dataset = get_conll_dataset()
    tokenized_dataset, label_list = tokenize_and_align_labels(dataset, tokenizer)

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
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()

