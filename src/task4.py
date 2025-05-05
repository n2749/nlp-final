import torch, numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer, TrainingArguments,
    AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling
)
import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

imdb = load_dataset("imdb")
bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")

def tok_bert(batch):
    return bert_tok(batch["text"], truncation=True, padding="max_length", max_length=128)

imdb_tok = imdb.map(tok_bert, batched=True, remove_columns=["text"])
imdb_tok = imdb_tok.rename_column("label", "labels")
imdb_tok.set_format("torch", columns=["input_ids","attention_mask","labels"])

bert_model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
).to(DEVICE)
collator = DataCollatorWithPadding(tokenizer=bert_tok)
metric_acc = evaluate.load("accuracy")
metric_prf = evaluate.load("precision")

def compute_metrics_bert(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    acc = metric_acc.compute(predictions=preds, references=labels)["accuracy"]
    pr = metric_prf.compute(predictions=preds, references=labels)["precision"]
    re = evaluate.load("recall").compute(predictions=preds, references=labels)["recall"]
    f1 = evaluate.load("f1").compute(predictions=preds, references=labels)["f1"]
    return {"accuracy":acc, "precision":pr, "recall":re, "f1":f1}

args_bert = TrainingArguments(
    output_dir="bert-out", num_train_epochs=2,
    per_device_train_batch_size=16, per_device_eval_batch_size=32,
    eval_strategy="epoch", logging_steps=100, save_strategy="no",
    learning_rate=2e-5, fp16=torch.cuda.is_available(),
    dataloader_num_workers=4, report_to="none"
)

trainer_bert = Trainer(
    model=bert_model, args=args_bert,
    train_dataset=imdb_tok["train"].shuffle(seed=42).select(range(5000)),
    eval_dataset=imdb_tok["test"].shuffle(seed=42).select(range(2000)),
    tokenizer=bert_tok, data_collator=collator,
    compute_metrics=compute_metrics_bert
)

trainer_bert.train()
bert_results = trainer_bert.evaluate()
print("BERT results:", bert_results)

shakes = load_dataset(
    "tiny_shakespeare",
    trust_remote_code=True
)

def save_local(split):
    path = f"{split}.txt"
    open(path, "w", encoding="utf-8").write("\n".join(shakes[split]["text"]))
    return path

train_file = save_local("train")
eval_file  = save_local("validation")

gpt_tok = AutoTokenizer.from_pretrained("gpt2")
gpt_tok.pad_token = gpt_tok.eos_token

train_ds = TextDataset(tokenizer=gpt_tok, file_path=train_file, block_size=128)
eval_ds  = TextDataset(tokenizer=gpt_tok, file_path=eval_file,  block_size=128)
coll_gpt = DataCollatorForLanguageModeling(tokenizer=gpt_tok, mlm=False)

gpt_model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)

args_gpt = TrainingArguments(
    output_dir="gpt2-out", num_train_epochs=3,
    per_device_train_batch_size=8, per_device_eval_batch_size=8,
    eval_strategy="epoch", logging_steps=200,
    save_strategy="epoch", learning_rate=5e-5,
    fp16=torch.cuda.is_available(), report_to="none"
)

trainer_gpt = Trainer(
    model=gpt_model, args=args_gpt,
    train_dataset=train_ds, eval_dataset=eval_ds,
    data_collator=coll_gpt, tokenizer=gpt_tok
)

trainer_gpt.train()
prompt = "To be, or not to be, "
inputs = gpt_tok(prompt, return_tensors="pt").to(DEVICE)
sample = gpt_model.generate(
    **inputs,
    max_length=100,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8,
    num_return_sequences=1
)
print("GPT-2 sample:\n", gpt_tok.decode(sample[0], skip_special_tokens=True))

from transformers import AutoModelForSeq2SeqLM
t5_tok = AutoTokenizer.from_pretrained("t5-small")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(DEVICE)

sent = "The quick brown fox jumps over the lazy dog."
input_text = "translate English to German: " + sent
inputs = t5_tok(input_text, return_tensors="pt").to(DEVICE)

outputs = t5_model.generate(
    **inputs,
    max_length=64,
    num_beams=4,
    early_stopping=True
)
print("T5 translation:\n", t5_tok.decode(outputs[0], skip_special_tokens=True))