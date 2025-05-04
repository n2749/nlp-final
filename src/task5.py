import spacy
from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt


def load_conll_dataset():
    dataset = load_dataset("conll2003")
    return dataset["validation"]


def extract_spacy_entities(nlp, tokens_list):
    all_preds = []
    for tokens in tokens_list:
        text = " ".join(tokens)
        doc = nlp(text)
        pred_labels = ["O"] * len(tokens)
        for ent in doc.ents:
            ent_tokens = ent.text.split()
            for i, token in enumerate(tokens):
                if token in ent_tokens:
                    pred_labels[i] = "B-" + ent.label_ if i == 0 else "I-" + ent.label_
        all_preds.append(pred_labels)
    return all_preds


def extract_bert_entities(pipeline_fn, texts):
    all_preds = []
    for text in texts:
        results = pipeline_fn(text)
        pred_labels = ["O"] * len(text.split())
        for ent in results:
            token_span = ent['word'].strip("Ġ").split()
            for i, token in enumerate(text.split()):
                if token in token_span:
                    pred_labels[i] = "B-" + ent['entity_group'] if i == 0 else "I-" + ent['entity_group']
        all_preds.append(pred_labels)
    return all_preds


def evaluate_predictions(y_true, y_pred, model_name):
    y_true_flat = [label for seq in y_true for label in seq]
    y_pred_flat = [label for seq in y_pred for label in seq]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_flat, y_pred_flat, average="weighted")
    print(f"{model_name} - Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
    return f1


def main():
    print("Loading data and models...")
    dataset = load_conll_dataset()
    tokens = dataset["tokens"]
    labels = dataset["ner_tags"]
    label_names = dataset.features["ner_tags"].feature.names
    gold_labels = [[label_names[l] for l in seq] for seq in labels]

    nlp_spacy = spacy.load("en_core_web_sm")
    nlp_bert = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

    print("\nEvaluating spaCy...")
    spacy_preds = extract_spacy_entities(nlp_spacy, tokens)
    spacy_f1 = evaluate_predictions(gold_labels, spacy_preds, "spaCy")

    print("\nEvaluating BERT...")
    texts = [" ".join(seq) for seq in tokens]
    bert_preds = extract_bert_entities(nlp_bert, texts)
    bert_f1 = evaluate_predictions(gold_labels, bert_preds, "BERT")

    plt.bar(["spaCy", "BERT"], [spacy_f1, bert_f1], color=["skyblue", "lightgreen"])
    plt.title("F1 Score Comparison for NER")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("img/task5-ner_f1_comparison.png")


if __name__ == "__main__":
    main()

