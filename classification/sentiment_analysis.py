import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

FINETUNED_DEBERTA_BASE_BEST_PATH = "classification/checkpoints/deberta-base-finetuned-best"

ASPECT_PROMPTS = {
    "camera": "camera",
    "battery": "battery",
    "display": "display",
    "price": "price",
    "software": "software",
    "design": "design",
    "performance": "performance",
    "charging": "charging",
    "storage": "storage",
}

ABSA_LABELS   = ["negative", "neutral", "positive"]
OVERALL_LABELS = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}

ASPECT_CATEGORIES = {"camera", "battery", "display", "price","software", "design", "performance", "charging", "storage"}

def rephrase_aspect(category):
    return ASPECT_PROMPTS.get(str(category).lower().strip(), str(category).lower().strip())

# To handle objective rows (skip sentiment classification pipeline, assign neutral)
def build_objective_rows(df_objective):
    df_obj = df_objective.copy()
    df_obj["aspect_input"] = None
    df_obj["pipeline"] = "skipped (objective)"
    df_obj["roberta_sentiment"] = None
    df_obj["roberta_confidence"] = None
    df_obj["deberta_base_finetuned_sentiment"] = None
    df_obj["deberta_base_finetuned_confidence"] = None
    df_obj["final_sentiment"] = "neutral"
    df_obj["final_confidence"] = None
    return df_obj

# Normal sentiment analysis (Twitter-RoBERTa)
def run_normal_sentiment_analysis(df_overall, roberta_pipe):
    if df_overall.empty:
        return df_overall

    texts = df_overall["text"].fillna("").astype(str).tolist()
    sentiments = []
    confidences = []

    total = len(texts)
    batch_size = 16

    for start in range(0, total, batch_size):
        batch = texts[start:start + batch_size]
        results = roberta_pipe(batch, truncation=True, max_length=512)

        for result in results:
            raw_label = result["label"].upper()

            # cardiffnlp model returns 'positive'/'negative'/'neutral' directly
            label = result["label"].lower()
            if label not in ["positive", "negative", "neutral"]:
                # fallback for LABEL_0/1/2 format
                label = OVERALL_LABELS.get(raw_label, "neutral")

            sentiments.append(label)
            confidences.append(round(result["score"], 4))

        print(f"  Overall: processed {min(start + batch_size, total)}/{total}")

    df_overall = df_overall.copy()
    df_overall["pipeline"] = "overall"
    df_overall["roberta_sentiment"] = sentiments
    df_overall["roberta_confidence"] = confidences
    
    # Fill ABSA columns with N/A for overall rows
    df_overall["aspect_input"] = "overall"
    df_overall["deberta_base_finetuned_sentiment"] = "N/A"
    df_overall["deberta_base_finetuned_confidence"] = None
    df_overall["final_sentiment"] = df_overall["roberta_sentiment"]
    df_overall["final_confidence"] = df_overall["roberta_confidence"]

    return df_overall

# 8. Ascpect-based sentiment analysis (DeBERTa)
def predict_absa(text, aspect, tokenizer, model, model_name="model"):
    try:
        inputs = tokenizer(
            text,
            aspect,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512 # may need adjust later
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1).squeeze()
        label_idx = torch.argmax(probs).item()

        return ABSA_LABELS[label_idx], round(probs[label_idx].item(), 4)

    except Exception as e:
        print(f"  {model_name} error: {e}")
        return "error", 0.0

def run_absa(df_aspect, finetuned_base_tokenizer, finetuned_base_model):

    if df_aspect.empty:
        return df_aspect

    df_aspect = df_aspect.copy()
    df_aspect["aspect_input"] = df_aspect["comment_category"].apply(rephrase_aspect)

    finetuned_base_sentiments, finetuned_base_confidences = [], []

    total = len(df_aspect)

    for i, (_, row) in enumerate(df_aspect.iterrows()):
        text = str(row["text"])
        aspect = str(row["aspect_input"])

        f_sent, f_conf = predict_absa(text, aspect, finetuned_base_tokenizer, finetuned_base_model, "DeBERTa-base-finetuned")
        finetuned_base_sentiments.append(f_sent)
        finetuned_base_confidences.append(f_conf)

        if (i + 1) % 10 == 0:
            print(f"  ABSA: processed {i + 1}/{total}")

    df_aspect["pipeline"] = "absa"
    df_aspect["deberta_base_finetuned_sentiment"] = finetuned_base_sentiments
    df_aspect["deberta_base_finetuned_confidence"] = finetuned_base_confidences
    df_aspect["roberta_sentiment"]  = "N/A"
    df_aspect["roberta_confidence"] = None
    df_aspect["final_sentiment"] = df_aspect["deberta_base_finetuned_sentiment"]
    df_aspect["final_confidence"] = df_aspect["deberta_base_finetuned_confidence"]

    return df_aspect


def run_sentiment_analysis(input_csv, output_csv):

    # 1. Load dataset
    df = pd.read_csv(input_csv)
    df = df[["comment_id", "text", "comment_category", "final_subjectivity_label"]].dropna()

    # 2. Split by subjectivity label
    subjective_mask = df["final_subjectivity_label"] == "Subjective"
    objective_mask = df["final_subjectivity_label"] == "Objective"

    df_subjective = df[subjective_mask].copy()
    df_objective = df[objective_mask].copy()

    print(f"Total comments: {len(df)}")
    print(f"Objective comments: {len(df_objective)} -> auto-assigned 'neutral'")
    print(f"Subjective comments: {len(df_subjective)} -> will run hybrid sentiment classification pipeline")

    # 3. Handle objective rows (skip sentiment classification pipeline, assign neutral)
    df_objective_results = build_objective_rows(df_objective)

    # 4. Split subjective rows into overall vs aspect
    overall_mask = df_subjective["comment_category"] == "overall"
    aspect_mask = df_subjective["comment_category"].isin(ASPECT_CATEGORIES)

    df_overall = df_subjective[overall_mask].copy()
    df_aspect = df_subjective[aspect_mask].copy()

    print(f"  Overall comments: {len(df_overall)}")
    print(f"  Aspect comments: {len(df_aspect)}")

    # 5. Load models
    print("Loading Twitter-RoBERTa (overall sentiment analysis)...")
    roberta_pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest")

    print("Loading fine-tuned DeBERTa-base ABSA model...")
    finetuned_base_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DEBERTA_BASE_BEST_PATH, use_fast=False)
    finetuned_base_model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_DEBERTA_BASE_BEST_PATH)
    finetuned_base_model.eval()

    # 6. Run hybrid sentiment analysis pipeline (for subjective comments)
    print("\nRunning normal sentiment analysis (Twitter-RoBERTa)...")
    df_overall_results = run_normal_sentiment_analysis(df_overall, roberta_pipe)

    print("\nRunning aspect-based sentiment analysis (DeBERTa base fine-tuned)...")
    df_aspect_results  = run_absa(df_aspect, finetuned_base_tokenizer, finetuned_base_model)

    # 7. Combine results and save
    FINAL_COLS = [
    "comment_id",
    "text",
    "comment_category",
    "final_subjectivity_label",
    "aspect_input",
    "pipeline",
    "roberta_sentiment",
    "roberta_confidence",
    "deberta_base_finetuned_sentiment",
    "deberta_base_finetuned_confidence",
    "final_sentiment",
    "final_confidence"
    ]

    df_combined = pd.concat([df_overall_results, df_aspect_results, df_objective_results], ignore_index=True)[FINAL_COLS]
    df_combined.to_csv(output_csv, index=False)
    print("\nResults saved")

    # 8. Output terminal summary
    print("\n" + "═" * 55)
    print("SUMMARY OF SENTIMENT ANALYSIS PIPELINE")
    print("═" * 55)

    print(f"\nTotal comments processed: {len(df_combined)}")
    print(f"Subjective - Overall (RoBERTa): {len(df_overall_results)}")
    print(f"Subjective - Aspect  (DeBERTa): {len(df_aspect_results)}")
    print(f"Objective - Skipped: {len(df_objective_results)}")

    print("\n-- Final Sentiment Distribution (all comments) --")
    print(df_combined["final_sentiment"].value_counts().to_string())

    print("\n-- Overall Comments - RoBERTa Distribution --")
    if not df_overall_results.empty:
        print(df_overall_results["roberta_sentiment"].value_counts().to_string())

    print("\n-- Aspect Comments — DeBERTa Distribution --")
    if not df_aspect_results.empty:
        print(df_aspect_results["deberta_base_finetuned_sentiment"].value_counts().to_string())

    print("\n-- Final Sentiment by Category --")
    print(pd.crosstab(df_combined["comment_category"], df_combined["final_sentiment"]).to_string())

    print("\n-- Average Confidence by Pipeline --")
    print(df_combined.groupby("pipeline")["final_confidence"].mean().round(4).to_string())

    return df_combined