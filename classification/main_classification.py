import pandas as pd
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix)

from text_preprocessing_v2 import run_text_preprocessing
from subjectivity_detection import run_subjectivity_detection
from sentiment_analysis import run_pipeline_1, run_pipeline_2, run_pipeline_3, run_pipeline_4, run_pipeline_5

RAW_CSV = "data/annotation_candidates.csv"
PREPROCESSED_CSV = "data/annotation_candidates_cleaned.csv"
SUBJECTIVITY_CSV = "data/subjectivity_detection_results.csv"

GROUND_TRUTH_CSV = "data/eval_test.csv"
GROUND_TRUTH_COL = "annotator_1 (ken)"
KEY_COLUMNS = ["comment_id"]
SENTIMENT_LABELS = ["negative", "neutral", "positive"]


# Separates subjective from objective comments (save to SUBJECTIVITY_CSV)
def run_subjectivity():
    input_text_col = "cleaned_comments"
    df = run_subjectivity_detection(PREPROCESSED_CSV, SUBJECTIVITY_CSV, input_text_col)
    print(f"  Saved {len(df)} rows -> {SUBJECTIVITY_CSV}")

# Evaluation (merges classfication pipeline predictions with manuually annotated evaluation dataset and print a full evaluation report)
def evaluate_run(run_label, pred_csv, pred_col="final_sentiment"):
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

    pred_df = pd.read_csv(pred_csv)
    pred_df = pred_df[KEY_COLUMNS + [pred_col]].copy()
 
    # Normalise: strip whitespace, lowercase
    for df, cols in [(gt_df, KEY_COLUMNS + [GROUND_TRUTH_COL]), (pred_df, KEY_COLUMNS + [pred_col])]:
        for col in cols:
            df[col] = df[col].astype(str).str.strip().str.lower()
 
    # Merge predictions with ground truth
    merged = pd.merge(pred_df, gt_df, on=KEY_COLUMNS, how="inner")
    print(f"  Matched rows: {len(merged)}")
    if merged.empty:
        print("  No matching rows — skipping evaluation.")
        return None
 
    eval_df = merged[[GROUND_TRUTH_COL, pred_col]].dropna()
    y_true  = eval_df[GROUND_TRUTH_COL]
    y_pred  = eval_df[pred_col]
 
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
 
 
def save_classification_summary(results, path="data/classification_evaluation_summary.csv"):
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    print(f"\n{'═' * 75}")
    print(f"  Classification summary saved -> {path}")
    print(f"{'═' * 75}")
    print(df.to_string(index=False))


def main():

    all_summaries = []

    # print("\n[1/3] Running text preprocessing...")
    # run_text_preprocessing()

    print("\n[2/3] Running subjectivity detection...")
    run_subjectivity()

    # Run sentiment analysis pipelines (pipelines 1 - 5) - FOR ABLATION STUDIES

    # ### ============== COMMENT OUT UNNECESSARY RUNS DURING TESTING =================

    print("\n[3/3] Running sentiment analysis pipeline...")

    # # RUN 1: RoBERTa only (all rows)
    # run_pipeline_1(input_csv = SUBJECTIVITY_CSV, output_csv = "data/sentiment_analysis_results_run1.csv")
    # s = evaluate_run("Run1_RoBERTa_only", "data/sentiment_analysis_results_run1.csv")
    # if s: 
    #     all_summaries.append(s)

    # # RUN 2: Text preprocessing + RoBERTa (all rows)
    # run_pipeline_2(input_csv = SUBJECTIVITY_CSV, output_csv = "data/sentiment_analysis_results_run2.csv")
    # s = evaluate_run("Run2_Preprocess+RoBERTa", "data/sentiment_analysis_results_run2.csv")
    # if s: 
    #     all_summaries.append(s)

    # # RUN 3: Text preprocessing + Subjectivity detection + RoBERTa (all rows)
    # run_pipeline_3(input_csv = SUBJECTIVITY_CSV, output_csv = "data/sentiment_analysis_results_run3.csv")
    # s = evaluate_run("Run3_Subjectivity+RoBERTa", "data/sentiment_analysis_results_run3.csv")
    # if s: 
    #     all_summaries.append(s)

    # RUN 4: Text preprocessing + Subjectivity detection + RoBERTa (overall) + DeBERTa-base (ABSA)
    run_pipeline_4(input_csv = SUBJECTIVITY_CSV, output_csv = "data/sentiment_analysis_results_run4.csv")
    s = evaluate_run("Run4_Hybrid_DeBERTa_base", "data/sentiment_analysis_results_run4.csv")
    if s: 
        all_summaries.append(s)

    # RUN 5: Text preprocessing + Subjectivity detection + RoBERTa (overall) + DeBERTa-base-finetuned (ABSA)
    run_pipeline_5(input_csv = SUBJECTIVITY_CSV, output_csv = "data/sentiment_analysis_results_run5.csv")
    s = evaluate_run("Run5_Hybrid_DeBERTa_finetuned", "data/sentiment_analysis_results_run5.csv")
    if s: 
        all_summaries.append(s)

    save_classification_summary(all_summaries)

if __name__ == "__main__":
    main()