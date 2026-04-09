import pandas as pd
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix)

# from classification.text_preprocessing import run_text_preprocessing
from subjectivity_detection import run_subjectivity_detection
from sarcasm_detection import run_sarcasm_detection
from sentiment_analysis import run_roberta_full, run_roberta_with_deberta_base, run_roberta_with_deberta_base_finetuned

RAW_CSV = "data/annotation_candidates.csv"
PREPROCESSED_CSV = "data/annotation_candidates_cleaned.csv"
SUBJECTIVITY_CSV = "data/subjectivity_detection_results.csv"
SARCASM_CSV = "data/sarcasm_detection_results.csv"

# Sentiment classification pipeline output CSVs
RUN1_CSV = "data/sentiment_analysis_results_run1.csv"
RUN2_CSV = "data/sentiment_analysis_results_run2.csv"
RUN3_CSV = "data/sentiment_analysis_results_run3.csv"
RUN4_CSV = "data/sentiment_analysis_results_run4.csv"
RUN5_CSV = "data/final_sentiment_analysis_results.csv"
TEST_RUN_CSV = "data/sentiment_analysis_results_test_run.csv"

# Summary outputs
LATENCY_XLSX = "data/pipeline_latency_summary.xlsx"
SUMMARY_XLSX = "data/classification_evaluation_summary.xlsx"

# Evaluation dataset
GROUND_TRUTH_CSV = "data/annotated_evaluation_set.csv"
GROUND_TRUTH_COL = "sentiment_label"
KEY_COLUMNS = ["comment_id"]
SENTIMENT_LABELS = ["negative", "neutral", "positive"]

ASPECT_CATEGORIES = {"camera", "battery", "display", "price", "software", "design", "performance", "charging", "storage"}
POLARITY_FLIP = {"positive": "negative", "negative": "positive", "neutral": "neutral"}


# Separates subjective from objective comments (save to SUBJECTIVITY_CSV)
def run_subjectivity(input_csv, output_csv, input_text_col = "cleaned_comments"):
    df, gronlp_elapsed_s, senticnet_elapsed_s = run_subjectivity_detection(input_csv, output_csv, input_text_col)
    print(f"  Saved {len(df)} rows -> {SUBJECTIVITY_CSV}")
    fallback_count = int(df["needs_senticnet_fallback"].sum())
    return [
        {"stage": "Subjectivity — GroNLP model", "records": len(df), "elapsed_s": gronlp_elapsed_s},
        {"stage": "Subjectivity — SenticNet fallback", "records": fallback_count, "elapsed_s": senticnet_elapsed_s},
    ]

# Detects sarcasm in subjective comments (save to SARCASM_CSV)
def run_sarcasm(input_csv, output_csv, input_text_col = "cleaned_comments"): 
    df, elapsed_s = run_sarcasm_detection(input_csv, output_csv, input_text_col)
    print(f"  Saved {len(df)} rows -> {SARCASM_CSV}")
    return {"stage": "Sarcasm detection", "records": len(df), "elapsed_s": elapsed_s}

# Derive ablation predictions from the final pipeline (Run 5) output
def derive_ablation_predictions(run5_csv, subjectivity_mode="hybrid", apply_sarcasm_flip=False):
    """
    subjectivity_mode:
      - "none" -> no subjectivity filtering (Run 2)
      - "model" -> use gronlp_subjectivity_label only (Run 3)
      - "hybrid" -> use final_subjectivity_label (model + SenticNet fallback) (Run 4)
    """

    df = pd.read_csv(run5_csv)

    if subjectivity_mode not in {"none", "model", "hybrid"}:
        raise ValueError(f"Unsupported subjectivity_mode: {subjectivity_mode}")

    if subjectivity_mode == "model":
        subj_col = "gronlp_subjectivity_label"
        df[subj_col] = df[subj_col].fillna("Objective").astype(str).str.strip()
    elif subjectivity_mode == "hybrid":
        subj_col = "final_subjectivity_label"
        df[subj_col] = df[subj_col].fillna("Objective").astype(str).str.strip()
    else:
        subj_col = None

    derived = []

    for _, row in df.iterrows():
        if subj_col is not None and str(row.get(subj_col, "Objective")).strip() != "Subjective":
            derived.append("neutral")
            continue

        cat = str(row.get("comment_category", "")).strip().lower()

        if cat in ASPECT_CATEGORIES:
            sentiment = str(row.get("deberta_base_finetuned_sentiment", "neutral")).strip().lower()
        else:
            sentiment = str(row.get("roberta_sentiment", "neutral")).strip().lower()

        if apply_sarcasm_flip:
            sarcasm_label = str(row.get("sarcasm_label", "")).strip()
            if sarcasm_label == "Sarcastic":
                sentiment = POLARITY_FLIP.get(sentiment, sentiment)

        derived.append(sentiment)

    df["derived_sentiment"] = derived
    return df[["comment_id", "derived_sentiment"]]


# Evaluation (merges classfication pipeline predictions with manuually annotated evaluation dataset and print a full evaluation report)
def evaluate_run(run_label, pred_csv=None, pred_col="final_sentiment", pred_df=None):
    """
    Metrics:
    Accuracy - agreement % (exact match across all 3 classes)
    Per-class Precision / Recall / F1 / Support (for each label - neu/neg/pos)
    Macro average - unweighted mean across classes
    Weighted average - weighted by support
    Confusion matrix - rows = true, cols = predicted
    """

    print("\n" + "=" * 75)
    print(f"  EVALUATION  |  {run_label}  |  column: '{pred_col}'")
    print("=" * 75)
 
    gt_df = pd.read_csv(GROUND_TRUTH_CSV)
    gt_df = gt_df[KEY_COLUMNS + [GROUND_TRUTH_COL]].copy()

    if pred_df is not None:
        pred_df_local = pred_df.copy()
    elif pred_csv is not None:
        pred_df_local = pd.read_csv(pred_csv)
    else:
        raise ValueError(f"evaluate_run({run_label!r}): must supply either pred_csv or pred_df.")

    required_pred_cols = KEY_COLUMNS + [pred_col]
    missing_cols = [col for col in required_pred_cols if col not in pred_df_local.columns]
    if missing_cols:
        raise KeyError(
            f"evaluate_run({run_label!r}): missing required prediction column(s): {missing_cols}. "
            f"Available columns: {list(pred_df_local.columns)}")
 
    pred_df_local = pred_df_local[required_pred_cols].copy()

    # Normalise: strip whitespace, lowercase
    for df_, cols in [
        (gt_df, KEY_COLUMNS + [GROUND_TRUTH_COL]),
        (pred_df_local, required_pred_cols),
    ]:
        for col in cols:
            df_[col] = df_[col].astype(str).str.strip().str.lower()

    # Merge predictions with ground truth
    merged = pd.merge(pred_df_local, gt_df, on=KEY_COLUMNS, how="inner")
    print(f"  Matched rows: {len(merged)}")
    if merged.empty:
        print("  No matching rows — skipping evaluation.")
        return None
 
    eval_df = merged[[GROUND_TRUTH_COL, pred_col]].dropna()
    y_true = eval_df[GROUND_TRUTH_COL]
    y_pred = eval_df[pred_col]
 
    # Accuracy (agreement %)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n  Accuracy (agreement %): {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"  Samples evaluated    : {len(eval_df)}")
 
    # Per-class metrics (Precission, Recall, F1, Support)
    labels = SENTIMENT_LABELS
    p_per, r_per, f_per, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    print(f"\n  {'Label':<14}{'Precision':>12}{'Recall':>10}{'F1':>10}{'Support':>10}")
    print("  " + "─" * 56)
    for label, p, r, f, s in zip(labels, p_per, r_per, f_per, support):
        print(f"  {label:<14}{p:>12.4f}{r:>10.4f}{f:>10.4f}{s:>10}")
 
    # Macro and Weighted averages
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    print("  " + "─" * 56)
    print(f"  {'macro avg':<14}{p_mac:>12.4f}{r_mac:>10.4f}{f_mac:>10.4f}")
    print(f"  {'weighted avg':<14}{p_wt:>12.4f}{r_wt:>10.4f}{f_wt:>10.4f}")
 
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index   =[f"true_{l}"  for l in labels], columns =[f"pred_{l}"  for l in labels])
    print("\n  Confusion Matrix:")
    print(cm_df.to_string(index=True))
 
    # Full sklearn report (for reference)
    print("\n  Full Classification Report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
 
    # Build summary dict
    summary = {
        "run": run_label,
        "pred_col": pred_col,
        "samples_evaluated": len(eval_df),
        "accuracy": round(accuracy, 4),
    }
    for lbl, p, r, f, s in zip(labels, p_per, r_per, f_per, support):
        summary[f"precision_{lbl}"] = round(p, 4)
        summary[f"recall_{lbl}"] = round(r, 4)
        summary[f"f1_{lbl}"] = round(f, 4)
    
    summary.update({
        "macro_precision": round(p_mac, 4),
        "macro_recall": round(r_mac, 4),
        "macro_f1": round(f_mac, 4),
        "weighted_precision": round(p_wt,  4),
        "weighted_recall": round(r_wt,  4),
        "weighted_f1": round(f_wt,  4),
    })
    return summary

# Build latency summary for Run 4 (pipeline 3) - final pipeline
def build_latency_summary(run_label, preprocessing_stages=None, sentiment_stages=None, note=""):
    preprocessing_stages = preprocessing_stages or []
    sentiment_stages = sentiment_stages or []

    rows = []
    all_stages = preprocessing_stages + sentiment_stages
 
    for entry in all_stages:
        elapsed = entry["elapsed_s"]
        records = entry["records"]
        rows.append({
            "run": run_label,
            "stage": entry["stage"],
            "records": records,
            "elapsed_s": elapsed,
            "records_per_s": round(records / elapsed, 4) if elapsed > 0 else None,
            "s_per_record": round(elapsed / records, 6) if records > 0 else None,
            "note": "",
        })
 
    # TOTAL row — sums time across all model stages for this run
    total_elapsed = sum(e["elapsed_s"] for e in all_stages)
    total_records = sum(e["records"]   for e in all_stages)
    rows.append({
        "run": run_label,
        "stage": "TOTAL (all model stages)",
        "records": total_records,
        "elapsed_s": round(total_elapsed, 4),
        "records_per_s": round(total_records / total_elapsed, 4) if total_elapsed > 0 else None,
        "s_per_record": round(total_elapsed / total_records, 6) if total_records > 0 else None,
        "note": "excludes objective rows (rule-assigned neutral, no model time)",
    })
    return rows

def save_latency(latency_rows, path=LATENCY_XLSX):
    latency_df = pd.DataFrame(latency_rows)
    tp_cols = ["run", "stage", "records", "elapsed_s", "records_per_s", "s_per_record", "note"]
    for c in tp_cols:
        if c not in latency_df.columns:
            latency_df[c] = None
    latency_df = latency_df[tp_cols]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        latency_df.to_excel(writer, sheet_name="Latency", index=False)
    print(f"  Latency saved -> {path}")
 
def save_summary(eval_results, path=SUMMARY_XLSX):
    eval_df = pd.DataFrame(eval_results)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        eval_df.to_excel(writer, sheet_name="Classification Metrics", index=False)
    print(f"  Summary saved -> {path}")
 

def main():

    RUN_ABLATIONS = True  # Set True to run Runs 1, 2, 3 alongside the main pipeline (pipeline 3)

    all_summaries = []

    # Cached preprocessing timing - populated when a stage runs, then referenced by each run that depends on it
    preproc_timing = {"subjectivity_senticnet": None, "sarcasm": None}


    # --- Preprocessing before sentiment classification (run once, then comment out) ---
    # print("\n[1/4] Running text preprocessing...")
    # run_text_preprocessing(input_csv=RAW_CSV, output_csv=PREPROCESSED_CSV)

    # print("\n[2/4] Running subjectivity detection (model + SenticNet)...")
    # preproc_timing["subjectivity_senticnet"] = run_subjectivity(input_csv=PREPROCESSED_CSV, output_csv=SUBJECTIVITY_CSV)

    # print("\n[3/4] Running sarcasm detection...")
    # preproc_timing["sarcasm"] = run_sarcasm(input_csv=SUBJECTIVITY_CSV, output_csv=SARCASM_CSV)


    # --- Final Sentiment Classification Pipeline: Pipeline 3 (Run 5 in ablation) ---
    print("\n[4/4] Running sentiment analysis pipeline...")

    # RoBERTa for general sentiment analysis + DeBERTa-base-finetuned ABSA + subjectivity (model + SenticNet fallback) + sarcasm detection
    _, timing = run_roberta_with_deberta_base_finetuned(input_csv=SARCASM_CSV, output_csv=RUN5_CSV)
    s = evaluate_run("Run5_RoBERTa_DeBERTaBase_finetuned_subjectivity_hybrid_sarcasm", pred_csv=RUN5_CSV)
    if s:
        all_summaries.append(s)
    preproc = (preproc_timing["subjectivity_senticnet"] or []) + ([preproc_timing["sarcasm"]] if preproc_timing["sarcasm"] else [])
    save_latency(build_latency_summary("Run5_RoBERTa_DeBERTaBase_finetuned_subjectivity_hybrid_sarcasm", preprocessing_stages=preproc, sentiment_stages=timing))


    # --- Ablation Runs (optional) ---
    if RUN_ABLATIONS:

        # RUN 1: Normal sentiment analysis
        # RoBERTa for general sentiment analysis on all preprocessed comments
        run_roberta_full(input_csv=SARCASM_CSV, output_csv=RUN1_CSV)
        s = evaluate_run("Run1_RoBERTa_all_preprocessed", pred_csv=RUN1_CSV)
        if s:
            all_summaries.append(s)

        # RUN 2: + ABSA (using deberta-base-finetuned)
        # RoBERTa for overall preprocessed comments + DeBERTa-base-finetuned ABSA, no subjectivity detection and no sarcasm detection (Derived from Pipeline 3/Run 4 output)
        predictions_run2 = derive_ablation_predictions(
            RUN5_CSV,
            subjectivity_mode="none",
            apply_sarcasm_flip=False,
        ).rename(columns={"derived_sentiment": "ablation_sentiment_run2"})
        predictions_run2.to_csv(RUN2_CSV, index=False)
        s = evaluate_run("Run2_RoBERTa_DeBERTaBase_finetuned_no_subjectivity_no_sarcasm", pred_csv=RUN2_CSV, pred_col="ablation_sentiment_run2")
        if s:
            all_summaries.append(s)

        # RUN 3: + Subjectivity detection using model only
        # RoBERTa for overall preprocessed comments + DeBERTa-base-finetuned ABSA, subjectivity detection using model only and no sarcasm detection (Derived from Pipeline 3/Run 4 output)
        predictions_run3 = derive_ablation_predictions(
            RUN5_CSV,
            subjectivity_mode="model",
            apply_sarcasm_flip=False,
        ).rename(columns={"derived_sentiment": "ablation_sentiment_run3"})
        predictions_run3.to_csv(RUN3_CSV, index=False)
        s = evaluate_run("Run3_RoBERTa_DeBERTaBase_finetuned_subjectivity_model_only_no_sarcasm", pred_csv=RUN3_CSV, pred_col="ablation_sentiment_run3")
        if s:
            all_summaries.append(s)

        # RUN 4: + Subjectivity detection using hybrid approach (model + SenticNet)
        # RoBERTa for overall preprocessed comments + DeBERTa-base-finetuned ABSA, subjectivity detection using model and senticnet and no sarcasm detection (Derived from Pipeline 3/Run 4 output)
        predictions_run4 = derive_ablation_predictions(
            RUN5_CSV,
            subjectivity_mode="hybrid",
            apply_sarcasm_flip=False,
        ).rename(columns={"derived_sentiment": "ablation_sentiment_run4"})
        predictions_run4.to_csv(RUN4_CSV, index=False)
        s = evaluate_run("Run4_RoBERTa_DeBERTaBase_finetuned_subjectivity_hybrid_no_sarcasm", pred_csv=RUN4_CSV, pred_col="ablation_sentiment_run4")
        if s:
            all_summaries.append(s)

        # TEST RUN (not part of ablation, for comparison between deberta-base vs deberta-base-finetuned)
        # Pipeline 2: RoBERTa for general sentiment analysis + DeBERTa-base ABSA + subjectivity (model + SenticNet fallback) + sarcasm detection
        run_roberta_with_deberta_base(input_csv=SARCASM_CSV, output_csv=TEST_RUN_CSV)
        s = evaluate_run("TestRun_RoBERTa_DeBERTaBase_subjectivity_hybrid_sarcasm", pred_csv=TEST_RUN_CSV)
        if s:
            all_summaries.append(s)

    save_summary(all_summaries)

if __name__ == "__main__":
    main()