import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# 1. Aspect rephrasing map
ASPECT_PROMPTS = {
    "camera": "photo and video quality",
    "battery": "battery life",
    "display": "display and user interface",
    "price": "price and value",
    "software": "software",
    "design": "phone design",
    "performance": "performance and speed",
    "charging": "charging",
    "storage": "storage capacity",
}

ABSA_LABELS   = ["negative", "neutral", "positive"]
OVERALL_LABELS = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}

ASPECT_CATEGORIES = {"camera", "battery", "display", "price","software", "design", "performance", "charging", "storage"}

def rephrase_aspect(category):
    return ASPECT_PROMPTS.get(str(category).lower().strip(), str(category).lower().strip())


# 2. Load dataset
df = pd.read_csv("data/subjectivity_detection_results.csv")
df = df.sample(n=500, random_state=42)
df = df[["comment_category", "text"]].dropna()

# Separate "overall" category from other specific categories/aspects
overall_mask = df["category"] == "overall"
aspect_mask  = df["category"].isin(ASPECT_CATEGORIES)

df_overall = df[overall_mask].copy()
df_aspect  = df[aspect_mask].copy()

print(f"Total comments:   {len(df)}")
print(f"Overall comments: {len(df_overall)}")
print(f"Aspect comments:  {len(df_aspect)}")


# 3. Load models from Hugging Face
print("Loading Twitter-RoBERTa (overall sentiment analysis)...")
roberta_pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest")

print("Loading DeBERTa-base ABSA model...")
base_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1", use_fast=False)
base_model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
base_model.eval()

print("Loading DeBERTa-large ABSA model...")
large_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-large-absa-v1.1", use_fast=False)
large_model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-large-absa-v1.1")
large_model.eval()


# 4. Overall sentiment analysis (Twitter-RoBERTa)
def run_overall_sentiment(df_overall, roberta_pipe):
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
    df_overall["deberta_base_sentiment"] = "N/A"
    df_overall["deberta_base_confidence"] = None
    df_overall["deberta_large_sentiment"] = "N/A"
    df_overall["deberta_large_confidence"] = None
    df_overall["deberta_models_agree"] = None

    df_overall["final_sentiment"] = df_overall["roberta_sentiment"]
    df_overall["final_confidence"] = df_overall["roberta_confidence"]

    return df_overall


# 5. Ascpect-based sentiment analysis (DeBERTa)
def predict_absa(text, aspect, tokenizer, model, model_name="model"):
    try:
        inputs = tokenizer(
            aspect,
            text,
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


def run_absa(df_aspect, base_tokenizer, base_model, large_tokenizer, large_model):
    if df_aspect.empty:
        return df_aspect

    df_aspect = df_aspect.copy()
    df_aspect["aspect_input"] = df_aspect["category"].apply(rephrase_aspect)

    base_sentiments,  base_confidences  = [], []
    large_sentiments, large_confidences = [], []

    total = len(df_aspect)

    for i, (_, row) in enumerate(df_aspect.iterrows()):
        text = str(row["text"])
        aspect = str(row["aspect_input"])

        b_sent, b_conf = predict_absa(text, aspect, base_tokenizer, base_model, "DeBERTa-base")
        l_sent, l_conf = predict_absa(text, aspect, large_tokenizer, large_model, "DeBERTa-large")

        base_sentiments.append(b_sent)
        base_confidences.append(b_conf)
        large_sentiments.append(l_sent)
        large_confidences.append(l_conf)

        if (i + 1) % 10 == 0:
            print(f"  ABSA: processed {i + 1}/{total}")

    df_aspect["pipeline"] = "absa"
    df_aspect["deberta_base_sentiment"] = base_sentiments
    df_aspect["deberta_base_confidence"] = base_confidences
    df_aspect["deberta_large_sentiment"] = large_sentiments
    df_aspect["deberta_large_confidence"] = large_confidences
    df_aspect["deberta_models_agree"] = (df_aspect["deberta_base_sentiment"] == df_aspect["deberta_large_sentiment"])

    df_aspect["roberta_sentiment"]  = "N/A"
    df_aspect["roberta_confidence"] = None

    df_overall["final_sentiment"] = df_aspect["deberta_large_sentiment"]
    df_overall["final_confidence"] = df_aspect["deberta_large_confidence"]

    return df_aspect


# 6. Run hybrid sentiment analysis pipeline
print("\nRunning overall sentiment (Twitter-RoBERTa)...")
df_overall_results = run_overall_sentiment(df_overall, roberta_pipe)

print("\nRunning ABSA (DeBERTa base + large)...")
df_aspect_results  = run_absa(df_aspect, base_tokenizer, base_model, large_tokenizer, large_model)


# 7. Combine results and save
df_combined = pd.concat([df_overall_results, df_aspect_results], ignore_index=True)
df_combined.to_csv("data/sentiment_analysis_results.csv", index=False)
df_combined = df_combined[[
        "text",
        "comment_category",
        "aspect_input",
        "pipeline",
        "roberta_sentiment",
        "roberta_confidence",
        "deberta_base_sentiment",
        "deberta_base_confidence",
        "deberta_large_sentiment",
        "deberta_large_confidence",
        "deberta_models_agree",
        "final_sentiment",
        "final_confidence",
    ]]
print("\nResults saved")


# 8. Output terminal summary
print("\n" + "═" * 55)
print("HYBRID SENTIMENT ANALYSIS PIPELINE — SUMMARY")
print("═" * 55)

print(f"\nTotal comments processed: {len(df_combined)}")
print(f"  → Overall (RoBERTa):    {len(df_overall_results)}")
print(f"  → Aspect  (DeBERTa):    {len(df_aspect_results)}")

print("\n── Final Sentiment Distribution (all comments) ──")
print(df_combined["final_sentiment"].value_counts().to_string())

print("\n── Overall Comments — RoBERTa Distribution ──")
if not df_overall_results.empty:
    print(df_overall_results["roberta_sentiment"].value_counts().to_string())
else:
    print("  No overall comments found.")

print("\n── Aspect Comments — DeBERTa-base Distribution ──")
if not df_aspect_results.empty:
    print(df_aspect_results["deberta_base_sentiment"].value_counts().to_string())
else:
    print("  No aspect comments found.")

print("\n── Aspect Comments — DeBERTa-large Distribution ──")
if not df_aspect_results.empty:
    print(df_aspect_results["deberta_large_sentiment"].value_counts().to_string())
else:
    print("  No aspect comments found.")

print("\n── ABSA Model Agreement Rate ──")
if not df_aspect_results.empty:
    agree_rate = df_aspect_results["models_agree"].mean()
    print(f"  DeBERTa-base vs large: {agree_rate:.1%}")
else:
    print("  No aspect comments found.")

print("\n── Final Sentiment by Category ──")
print(pd.crosstab(df_combined["category"], df_combined["final_sentiment"]).to_string())

print("\n── Average Confidence by Pipeline ──")
print(df_combined.groupby("pipeline")["final_confidence"].mean().round(4).to_string())

print("\n" + "═" * 55)