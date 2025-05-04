from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer as GPTTrainer, TrainingArguments as GPTTrainingArguments
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import DataCollatorWithPadding
from transformers import MarianMTModel, MarianTokenizer
import evaluate
import numpy as np
import matplotlib.pyplot as plt
import torch

BERT_PLOT_FILENAME = "img/task4-bert.png"


def get_imdb(tokenizer, sample_train_size=5000, sample_test_size=1000):
    dataset = load_dataset("imdb")

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(sample_train_size))
    test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(sample_test_size))

    return train_dataset, test_dataset


def get_bert_model():
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)


def plot_metrics(trainer):
    if trainer.state.log_history:
        train_accuracies = [log.get("eval_accuracy") for log in trainer.state.log_history if "eval_accuracy" in log]
        epochs = list(range(1, len(train_accuracies) + 1))

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_accuracies, marker='o', label="Validation Accuracy")
        plt.title("Validation Accuracy per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(BERT_PLOT_FILENAME)


def bert():
    model, tokenizer = get_bert_model()
    train, test = get_imdb(tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./bert-imdb",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    plot_metrics(trainer)


def gpt():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path="shakespeare.txt",
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = GPTTrainingArguments(
        output_dir="./gpt-finetuned",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs"
    )

    trainer = GPTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    input_text = "To be, or not to be"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\nGenerated text:\n", generated_text)


def t5_translation():
    model_name = "Helsinki-NLP/opus-mt-ru-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    original_text = "Привет, как дела?"
    inputs = tokenizer(original_text, return_tensors="pt", padding=True)

    translated_ids = model.generate(**inputs)
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)

    print("\nOriginal Russian:\n", original_text)
    print("\nTranslated English:\n", translated_text)



def main():
    # Uncomment to run a model
    bert()
    # gpt()
    # t5_translation()


if __name__ == "__main__":
    main()

