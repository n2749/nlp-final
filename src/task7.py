import torch
import random
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

ds = load_dataset("imdb")
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

ds_train = ds["train"].map(tokenize_fn, batched=True, remove_columns=["text"])
ds_test  = ds["test"].map(tokenize_fn,  batched=True, remove_columns=["text"])

ds_train.set_format(type="torch", columns=["input_ids","attention_mask","label"])
ds_test .set_format(type="torch", columns=["input_ids","attention_mask","label"])

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=2
).to(DEVICE)

def compute_metrics(p):
    preds  = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="binary"),
    }

training_args = TrainingArguments(
    output_dir="bert-sentiment",
    num_train_epochs=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",
    logging_steps=100,
    save_strategy="no",
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=4,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
bert_results = trainer.evaluate()
print("BERT results:", bert_results)

bert_preds  = np.argmax(trainer.predict(ds_test).predictions, axis=1)
bert_labels = np.array(ds_test["label"])

class IMDBDataset(Dataset):
    def __init__(self, hf_ds):
        self.ids    = hf_ds["input_ids"]
        self.mask   = hf_ds["attention_mask"]
        self.labels = hf_ds["label"]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            "input_ids":      self.ids[idx],
            "attention_mask": self.mask[idx],
            "labels":         self.labels[idx],
        }

train_loader = DataLoader(
    IMDBDataset(ds_train),
    batch_size=128,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)
test_loader = DataLoader(
    IMDBDataset(ds_test),
    batch_size=128,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

class BiLSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim,
                                  padding_idx=tokenizer.pad_token_id)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim,
                             batch_first=True, bidirectional=True)
        self.fc    = nn.Linear(hidden_dim*2, 2)
    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
        out, _ = self.lstm(x)
        fwd = out[:, -1, :out.size(2)//2]
        bwd = out[:,  0, out.size(2)//2:]
        return self.fc(torch.cat([fwd, bwd], dim=1))

lstm_model = BiLSTMSentiment(tokenizer.vocab_size).to(DEVICE)
optimizer  = optim.Adam(lstm_model.parameters(), lr=1e-3)
criterion  = nn.CrossEntropyLoss()

for epoch in range(1, 4):
    lstm_model.train()
    total_loss = 0
    for batch in train_loader:
        ids    = batch["input_ids"].to(DEVICE)
        mask   = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        optimizer.zero_grad()
        logits = lstm_model(ids, mask)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[LSTM] Epoch {epoch} Loss: {total_loss/len(train_loader):.4f}")

lstm_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        ids    = batch["input_ids"].to(DEVICE)
        mask   = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        logits = lstm_model(ids, mask)
        preds  = torch.argmax(logits, dim=1)
        all_preds .extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

lstm_acc = accuracy_score(all_labels, all_preds)
lstm_f1  = f1_score(all_labels, all_preds, average="binary")
print(f"[LSTM] Accuracy: {lstm_acc:.4f}, F1: {lstm_f1:.4f}")
print("BERT Confusion Matrix:")
print(confusion_matrix(bert_labels, bert_preds))
print(classification_report(bert_labels, bert_preds, digits=4))
print("\nLSTM Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
print(classification_report(all_labels, all_preds, digits=4))