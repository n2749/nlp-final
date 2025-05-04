import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from tqdm import tqdm


class IMDBDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Sigmoid()

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(embedded)
        output = self.fc(hidden[-1])
        return self.softmax(output.squeeze(1))


def compute_and_plot_metrics(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{model_name} Accuracy: {acc:.4f}")
    print(f"{model_name} F1-score: {f1:.4f}")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["neg", "pos"], yticklabels=["neg", "pos"])
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.show()


def plot_epoch_accuracies(accuracies, model_name):
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o')
    plt.title(f"{model_name} Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{model_name.lower().replace(' ', '_')}_epoch_accuracy.png")
    plt.show()


def run_bert():
    dataset = load_dataset("imdb")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_fn(example):
        return tokenizer(example["text"], truncation=True, padding=True)

    tokenized = dataset.map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./bert-imdb",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        logging_dir="./logs",
        logging_steps=100,
        save_strategy="no"
    )

    accuracies = []

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        accuracies.append(acc)
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"].shuffle(seed=42).select(range(5000)),
        eval_dataset=tokenized["test"].shuffle(seed=42).select(range(1000)),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    predictions = trainer.predict(tokenized["test"].select(range(1000)))
    preds = np.argmax(predictions.predictions, axis=1)
    compute_and_plot_metrics(predictions.label_ids, preds, model_name="BERT")
    plot_epoch_accuracies(accuracies, model_name="BERT")


def run_lstm():
    dataset = load_dataset("imdb")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    tokenized = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=256), batched=True)
    vocab_size = tokenizer.vocab_size

    train_dataset = IMDBDataset(tokenized["train"].shuffle(seed=42).select(range(5000)), dataset["train"].shuffle(seed=42).select(range(5000))["label"])
    test_dataset = IMDBDataset(tokenized["test"].shuffle(seed=42).select(range(1000)), dataset["test"].shuffle(seed=42).select(range(1000))["label"])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = LSTMClassifier(vocab_size=vocab_size, embedding_dim=128, hidden_dim=128, output_dim=1)
    criterion = nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)

    lstm_accuracies = []
    model.train()
    for epoch in range(2):
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            output = model(batch['input_ids'])
            loss = criterion(output, batch['labels'].float())
            loss.backward()
            optimizer.step()

        model.eval()
        epoch_preds = []
        epoch_labels = []
        with torch.no_grad():
            for batch in test_loader:
                output = model(batch['input_ids'])
                preds = (output > 0.5).int()
                epoch_preds.extend(preds.tolist())
                epoch_labels.extend(batch['labels'].tolist())

        acc = accuracy_score(epoch_labels, epoch_preds)
        lstm_accuracies.append(acc)

    compute_and_plot_metrics(epoch_labels, epoch_preds, model_name="LSTM")
    plot_epoch_accuracies(lstm_accuracies, model_name="LSTM")


def main():
    run_bert()
    run_lstm()


if __name__ == "__main__":
    main()

