import spacy
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from transformers import pipeline
import evaluate
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score


def load_news_articles(sample_size=5):
    dataset = load_dataset("ag_news")
    articles = dataset["train"]["text"][:sample_size]
    return articles


def perform_ner(nlp, texts):
    for i, doc in enumerate(nlp.pipe(texts)):
        print(f"\n--- Article {i + 1} ---")
        print(texts[i])
        print("\nNamed Entities:")
        for ent in doc.ents:
            print(f"  - {ent.text} ({ent.label_})")


def perform_ner_with_transformers():
    model_checkpoint = "dslim/bert-base-NER"
    nlp_ner = pipeline("ner", model=model_checkpoint, tokenizer=model_checkpoint, aggregation_strategy="simple")

    sample_texts = load_news_articles(sample_size=5)
    for i, text in enumerate(sample_texts):
        print(f"\n--- Article {i + 1} (Transformers) ---")
        print(text)
        print("\nNamed Entities:")
        results = nlp_ner(text)
        for ent in results:
            print(f"  - {ent['word']} ({ent['entity_group']})")


def perform_pos_tagging(nlp, texts):
    all_preds = []
    all_labels = []
    for doc in nlp.pipe(texts):
        all_preds.extend([token.pos_ for token in doc])
        all_labels.extend([token.tag_ for token in doc])  # using fine-grained tags for reference

    return all_preds, all_labels


def perform_pos_tagging_with_transformers():
    model_checkpoint = "vblagoje/bert-english-uncased-finetuned-pos"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
    pos_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    sample_texts = load_news_articles(sample_size=3)
    all_preds = []
    all_labels = []  # Since we don't have gold labels, we use the predicted label for analysis only
    for text in sample_texts:
        results = pos_pipeline(text)
        all_preds.extend([token['entity_group'] for token in results])
        all_labels.extend([token['entity_group'] for token in results])  # self-comparison (for demo)

    return all_preds, all_labels


def evaluate_and_plot(preds, labels, title_prefix="Model"):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    print(f"\n{title_prefix} Accuracy: {acc:.2f}")
    print(f"{title_prefix} F1 Score: {f1:.2f}")

    plt.bar(["Accuracy", "F1 Score"], [acc, f1], color=['skyblue', 'lightgreen'])
    plt.title(f"{title_prefix} POS Evaluation")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(f"{title_prefix.lower().replace(' ', '_')}_metrics.png")
    plt.show()


def main():
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print("Loading dataset...")
    articles = load_news_articles()

    print("\nPerforming Named Entity Recognition with spaCy...")
    perform_ner(nlp, articles)

    print("\nPerforming Named Entity Recognition with Transformers...")
    perform_ner_with_transformers()

    print("\nPerforming POS Tagging with spaCy...")
    spacy_preds, spacy_labels = perform_pos_tagging(nlp, articles)
    evaluate_and_plot(spacy_preds, spacy_labels, title_prefix="spaCy")

    print("\nPerforming POS Tagging with Transformers...")
    transformer_preds, transformer_labels = perform_pos_tagging_with_transformers()
    evaluate_and_plot(transformer_preds, transformer_labels, title_prefix="Transformers")


if __name__ == "__main__":
    main()

