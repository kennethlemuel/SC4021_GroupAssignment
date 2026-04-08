import pandas as pd
import torch
from tqdm.auto import tqdm
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

def load_base_columns(input_csv):
    df = pd.read_csv(input_csv)
    df = df[["comment_id", "text", "cleaned_comments", "comment_category", "final_subjectivity_label"]].dropna()
    return df

def save_df(df, output_csv, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df[cols].to_csv(output_csv, index=False)
    print(f"\n  Saved {len(df)} rows → {output_csv}")
    return df[cols]

def print_run_header(run_no, description):
    print(f"\n{'━' * 65}")
    print(f"  Run {run_no}: {description}")
    print(f"{'━' * 65}")

def print_summary(df_combined, df_overall=None, df_aspect=None, df_objective=None):
    print("\n" + "═" * 45)
    print("SENTIMENT ANALYSIS PIPELINE SUMMARY")
    print("═" * 45)

    # Counts
    print(f"Total: {len(df_combined)} | Overall: {len(df_overall) if df_overall is not None else 0} | Aspect: {len(df_aspect) if df_aspect is not None else 0} | Skipped: {len(df_objective) if df_objective is not None else 0}")

    # Final sentiment distribution
    print("\nFinal:", df_combined["final_sentiment"].value_counts().to_dict())

    # RoBERTa
    if df_overall is not None and not df_overall.empty:
        print("RoBERTa:", df_overall["roberta_sentiment"].value_counts().to_dict())

    # DeBERTa
    if df_aspect is not None and not df_aspect.empty:
        for col in ["deberta_base_sentiment", "deberta_base_finetuned_sentiment"]:
            if col in df_aspect.columns:
                print(f"{col}:", df_aspect[col].value_counts().to_dict())


# To handle objective rows (skip sentiment classification pipeline, assign neutral)
def build_objective_rows(df_objective, extra_null_cols=None):
    df = df_objective.copy()
    df["pipeline"] = "skipped (objective)"
    df["final_sentiment"] = "neutral"
    df["final_confidence"] = None
    if extra_null_cols:
        for col in extra_null_cols:
            if col not in df.columns:
                df[col] = None
    return df

# ----- Load models -----
def load_roberta():
    print("  Loading Twitter-RoBERTa...")
    return pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest")
 
def load_deberta_base():
    print("  Loading DeBERTa-v3-base ABSA model...")
    tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1", use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
    model.eval()
    return tokenizer, model
 
def load_deberta_finetuned():
    print("  Loading fine-tuned DeBERTa-v3-base ABSA model...")
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DEBERTA_BASE_BEST_PATH, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_DEBERTA_BASE_BEST_PATH)
    model.eval()
    return tokenizer, model

# ----- To run normal sentiment analysis (Twitter-Roberta) -----

# Runs RoBERTa on one batch only, returns (sentiments, confidences)
def run_roberta(batch_texts, roberta_pipe):
    results = roberta_pipe(batch_texts, truncation=True, max_length=512)

    sentiments, confidences = [], []

    for r in results:
        label = r["label"].lower()
        if label not in ["positive", "negative", "neutral"]:
            label = OVERALL_LABELS.get(r["label"].upper(), "neutral")

        sentiments.append(label)
        confidences.append(round(r["score"], 4))

    return sentiments, confidences

# To attach roberta_sentiment/roberta_confidence to every row in df
def apply_roberta(df, roberta_pipe, tag="RoBERTa", text_col="text", batch_size=16):
    df = df.copy()
    texts = df[text_col].fillna("").astype(str).tolist()

    all_sentiments = []
    all_confidences = []

    for start in tqdm(range(0, len(texts), batch_size), desc=tag, unit="batch"):
        batch_texts = texts[start:start + batch_size]
        sentiments, confidences = run_roberta(batch_texts, roberta_pipe)

        all_sentiments.extend(sentiments)
        all_confidences.extend(confidences)

    df["roberta_sentiment"] = all_sentiments
    df["roberta_confidence"] = all_confidences
    return df

# ----- To run aspect-based sentiment analysis (DeBERTa-base, DeBERTa-base-finetuned) -----

# Runs ABSA on one batch only, returns (sentiments, confidences)
def run_absa(batch_texts, batch_aspects, tokenizer, model, model_name="model"):
    try:
        device = next(model.parameters()).device

        inputs = tokenizer(
            batch_texts,
            batch_aspects,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512 # may need adjust later
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)
        pred_indices = torch.argmax(probs, dim=-1)

        sentiments = []
        confidences = []

        for i in range(len(batch_texts)):
            label_idx = pred_indices[i].item()
            conf = probs[i][label_idx].item()

            sentiments.append(ABSA_LABELS[label_idx])
            confidences.append(round(conf, 4))

        return sentiments, confidences

    except Exception as e:
        print(f"    {model_name} batch error: {e}")
        return ["error"] * len(batch_texts), [0.0] * len(batch_texts)

# To attach sent_col/conf_col to df
def apply_absa(df, tokenizer, model, sent_col, conf_col, model_name="DeBERTa", text_col="text", batch_size=16):
    df = df.copy()

    if "aspect_input" not in df.columns:
        df["aspect_input"] = df["comment_category"].apply(rephrase_aspect)

    texts = df[text_col].fillna("").astype(str).tolist()
    aspects = df["aspect_input"].fillna("").astype(str).tolist()

    all_sentiments = []
    all_confidences = []

    for start in tqdm(range(0, len(df), batch_size), desc=f"{model_name} ABSA", unit="batch"):
        batch_texts = texts[start:start + batch_size]
        batch_aspects = aspects[start:start + batch_size]

        sentiments, confidences = run_absa(
            batch_texts,
            batch_aspects,
            tokenizer,
            model,
            model_name
        )

        all_sentiments.extend(sentiments)
        all_confidences.extend(confidences)

    df[sent_col] = all_sentiments
    df[conf_col] = all_confidences
    return df

# ----- Runs -----
# RUN 1: RoBERTa only
def run_pipeline_1(input_csv, output_csv):

    print_run_header(1, "RoBERTa only - all rows")
    df = load_base_columns(input_csv)
    print(f"  Total rows: {len(df)}")
 
    roberta_pipe = load_roberta()
    print("  Running RoBERTa on all rows...")
    df = apply_roberta(df, roberta_pipe, "RoBERTa-all", text_col="text")
 
    df["pipeline"] = "roberta_only"
    df["final_sentiment"] = df["roberta_sentiment"]
    df["final_confidence"] = df["roberta_confidence"]
 
    COLS = ["comment_id", "text", "comment_category", "final_subjectivity_label", "pipeline", "roberta_sentiment", "roberta_confidence", "final_sentiment", "final_confidence"]
    print_summary(df_combined=df, df_overall=df)
    return save_df(df, output_csv, COLS)

# RUN 2: Text preprocessing + RoBERTa
def run_pipeline_2(input_csv, output_csv):

    print_run_header(2, "Text preprocessing + RoBERTa - all rows")
    df = load_base_columns(input_csv)
    print(f"  Total rows: {len(df)}")
 
    roberta_pipe = load_roberta()
    print("  Running RoBERTa on preprocessed rows...")
    df = apply_roberta(df, roberta_pipe, "RoBERTa-preprocessed", text_col="cleaned_comments")
 
    df["pipeline"] = "preprocess+roberta"
    df["final_sentiment"] = df["roberta_sentiment"]
    df["final_confidence"] = df["roberta_confidence"]
 
    COLS = ["comment_id", "text", "cleaned_comments", "comment_category", "final_subjectivity_label", "pipeline", "roberta_sentiment", "roberta_confidence", "final_sentiment", "final_confidence"]
    print_summary(df_combined=df, df_overall=df)
    return save_df(df, output_csv, COLS)

# RUN 3: Text preprocessing + Subjectivity detection + RoBERTa
def run_pipeline_3(input_csv, output_csv):

    print_run_header(3, "Text preprocessing + subjectivity detection + RoBERTa (subjective only)")
    df = load_base_columns(input_csv)
 
    df_subj = df[df["final_subjectivity_label"] == "Subjective"].copy()
    df_obj = df[df["final_subjectivity_label"] == "Objective"].copy()
    print(f"  Total: {len(df)}  |  Subjective: {len(df_subj)}  |  Objective: {len(df_obj)}")
 
    # Objective -> neutral (no model)
    df_obj_out = build_objective_rows(df_obj, extra_null_cols=["roberta_sentiment", "roberta_confidence"])
 
    # Subjective -> RoBERTa
    roberta_pipe = load_roberta()
    print("  Running RoBERTa on subjective rows...")
    df_subj = apply_roberta(df_subj, roberta_pipe, "RoBERTa-subjective", text_col="cleaned_comments")
    df_subj["pipeline"] = "preprocess+subjectivity+roberta"
    df_subj["final_sentiment"] = df_subj["roberta_sentiment"]
    df_subj["final_confidence"] = df_subj["roberta_confidence"]
 
    COLS = ["comment_id", "text", "cleaned_comments", "comment_category", "final_subjectivity_label", "pipeline", "roberta_sentiment", "roberta_confidence", "final_sentiment", "final_confidence"]
    df_out = pd.concat([df_subj, df_obj_out], ignore_index=True)
    print_summary(df_combined=df_out, df_overall=df_subj, df_objective=df_obj_out)
    return save_df(df_out, output_csv, COLS)

# RUN 4: Text preprocessing + Subjectivity detection + RoBERTa (overall) + DeBERTa-base (ABSA)
def run_pipeline_4(input_csv, output_csv):

    print_run_header(4, "Full hybrid: RoBERTa (overall) + DeBERTa-base ABSA (aspect)")
    df = load_base_columns(input_csv)
 
    df_subj = df[df["final_subjectivity_label"] == "Subjective"].copy()
    df_obj = df[df["final_subjectivity_label"] == "Objective"].copy()
    df_overall = df_subj[df_subj["comment_category"] == "overall"].copy()
    df_aspect  = df_subj[df_subj["comment_category"].isin(ASPECT_CATEGORIES)].copy()
    print(f"  Total: {len(df)}  |  Objective: {len(df_obj)} |  Overall: {len(df_overall)} |  Aspect: {len(df_aspect)}")
 
    roberta_pipe = load_roberta()
    base_tokenizer, base_model = load_deberta_base()
 
    # Objective -> neutral
    df_obj_out = build_objective_rows(df_obj, extra_null_cols=["roberta_sentiment", "roberta_confidence", "aspect_input", "deberta_base_sentiment", "deberta_base_confidence"])
 
    # Overall -> RoBERTa
    print("  Running RoBERTa on overall rows...")
    df_overall = apply_roberta(df_overall, roberta_pipe, "RoBERTa-overall", text_col="cleaned_comments")
    df_overall["pipeline"] = "preprocess+subjectivity+roberta"
    df_overall["aspect_input"] = "overall"
    df_overall["deberta_base_sentiment"] = "N/A"
    df_overall["deberta_base_confidence"] = None
    df_overall["final_sentiment"] = df_overall["roberta_sentiment"]
    df_overall["final_confidence"] = df_overall["roberta_confidence"]
 
    # Aspect -> DeBERTa-base ABSA
    df_aspect["aspect_input"] = df_aspect["comment_category"].apply(rephrase_aspect)
    print("  Running DeBERTa-base ABSA on aspect rows...")
    df_aspect = apply_absa(df_aspect, base_tokenizer, base_model, "deberta_base_sentiment", "deberta_base_confidence", "DeBERTa-base", text_col="cleaned_comments")
    df_aspect["pipeline"] = "preprocess+subjectivity+absa_base"
    df_aspect["roberta_sentiment"] = "N/A"
    df_aspect["roberta_confidence"] = None
    df_aspect["final_sentiment"] = df_aspect["deberta_base_sentiment"]
    df_aspect["final_confidence"] = df_aspect["deberta_base_confidence"]
 
    COLS = ["comment_id", "text", "cleaned_comments", "comment_category", "final_subjectivity_label", "pipeline", "roberta_sentiment", "roberta_confidence", "aspect_input", "deberta_base_sentiment", "deberta_base_confidence", "final_sentiment", "final_confidence"]
    df_out = pd.concat([df_overall, df_aspect, df_obj_out], ignore_index=True)
    print_summary(df_combined=df_out, df_overall=df_overall, df_aspect=df_aspect, df_objective=df_obj_out)
    return save_df(df_out, output_csv, COLS)

# RUN 5: Text preprocessing + Subjectivity detection + RoBERTa (overall) + DeBERTa-base-finetuned (ABSA)
def run_pipeline_5(input_csv, output_csv):

    print_run_header(5, "Full hybrid: RoBERTa (overall) + DeBERTa-base-finetuned ABSA (aspect)")
    df = load_base_columns(input_csv)
 
    df_subj = df[df["final_subjectivity_label"] == "Subjective"].copy()
    df_obj = df[df["final_subjectivity_label"] == "Objective"].copy()
    df_overall = df_subj[df_subj["comment_category"] == "overall"].copy()
    df_aspect = df_subj[df_subj["comment_category"].isin(ASPECT_CATEGORIES)].copy()
    print(f"  Total: {len(df)}  |  Objective: {len(df_obj)}  |  Overall: {len(df_overall)}  |  Aspect: {len(df_aspect)}")
 
    roberta_pipe = load_roberta()
    finetuned_base_tokenizer, finetuned_base_model  = load_deberta_finetuned()
 
    # Objective -> neutral
    df_obj_out = build_objective_rows(df_obj, extra_null_cols=["roberta_sentiment", "roberta_confidence", "aspect_input", "deberta_base_finetuned_sentiment", "deberta_base_finetuned_confidence"])
 
    # Overall -> RoBERTa
    print("  Running RoBERTa on overall rows...")
    df_overall = apply_roberta(df_overall, roberta_pipe, "RoBERTa-overall", text_col="cleaned_comments")
    df_overall["pipeline"] = "preprocessing+subjectivity+roberta"
    df_overall["aspect_input"] = "overall"
    df_overall["deberta_base_finetuned_sentiment"] = "N/A"
    df_overall["deberta_base_finetuned_confidence"] = None
    df_overall["final_sentiment"] = df_overall["roberta_sentiment"]
    df_overall["final_confidence"] = df_overall["roberta_confidence"]
 
    # Aspect → fine-tuned DeBERTa-base ABSA
    df_aspect["aspect_input"] = df_aspect["comment_category"].apply(rephrase_aspect)
    print("  Running DeBERTa-base-finetuned ABSA on aspect rows...")
    df_aspect = apply_absa(df_aspect, finetuned_base_tokenizer, finetuned_base_model, "deberta_base_finetuned_sentiment", "deberta_base_finetuned_confidence", "DeBERTa-base-finetuned", text_col="cleaned_comments")
    df_aspect["pipeline"] = "preprocessing+subjectivity+absa_finetuned"
    df_aspect["roberta_sentiment"] = "N/A"
    df_aspect["roberta_confidence"] = None
    df_aspect["final_sentiment"] = df_aspect["deberta_base_finetuned_sentiment"]
    df_aspect["final_confidence"] = df_aspect["deberta_base_finetuned_confidence"]
 
    COLS = ["comment_id", "text", "cleaned_comments", "comment_category", "final_subjectivity_label", "pipeline", "roberta_sentiment", "roberta_confidence", "aspect_input", "deberta_base_finetuned_sentiment", "deberta_base_finetuned_confidence", "final_sentiment", "final_confidence"]
    df_out = pd.concat([df_overall, df_aspect, df_obj_out], ignore_index=True)
    print_summary(df_combined=df_out, df_overall=df_overall, df_aspect=df_aspect, df_objective=df_obj_out)
    return save_df(df_out, output_csv, COLS)

