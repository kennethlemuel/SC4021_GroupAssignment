import pandas as pd
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix)

from text_preprocessing_v2 import run_text_preprocessing
from subjectivity_detection import run_subjectivity_detection
from sentiment_analysis import run_sentiment_analysis

def main():

    print("\n[1/4] Running text preprocessing...")
    # df_preprocessed = run_text_preprocessing()
    # print(f"Preprocessed rows: {len(df_preprocessed)}")


    # print("\n[2/4] Running subjectivity detection...")
    # subjectivity_input_csv = "data/annotation_candidates_cleaned.csv"
    # subjectibity_output_csv = "data/subjectivity_detection_results.csv"
    # text_col = "cleaned_comments"
    # df_subjectivity = run_subjectivity_detection(subjectivity_input_csv, subjectibity_output_csv, text_col)
    # print(f"Number of rows after subjectivity detection: {len(df_subjectivity)}")


    print("\n[3/4] Running sentiment analysis...")
    sentiment_input_csv = "data/subjectivity_detection_results.csv"
    sentiment_output_csv = "data/sentiment_analysis_results.csv"
    df_sentiment = run_sentiment_analysis(sentiment_input_csv, sentiment_output_csv)
    print(f"Number of rows after sentiment classification: {len(df_sentiment)}")

    print(f"\nFinal output saved to: {sentiment_input_csv}")


    print("\n[4/4] Running evaluation...")

    ground_truth_csv = "data/eval_test.csv"
    sentiment_analysis_csv = "data/sentiment_analysis_results.csv"
    key_columns = ["comment_id"]
    ground_truth_col = "annotator_1 (ken)"
    sentiment_prediction_cols = ["final_base_sentiment","final_finetuned_sentiment","final_large_sentiment"]

    # Load ground truth (annotated labels)
    gt_df = pd.read_csv(ground_truth_csv)

    # Load model predictions for sentiment analysis
    pred_df = pd.read_csv(sentiment_analysis_csv)

    # Keep only needed columns
    gt_df = gt_df[key_columns + [ground_truth_col]].copy()
    pred_df = pred_df[key_columns + sentiment_prediction_cols].copy()

    # Clean text (important for matching)
    for col in key_columns + [ground_truth_col] + sentiment_prediction_cols:
        if col in pred_df.columns:
            pred_df[col] = pred_df[col].astype(str).str.strip()
        if col in gt_df.columns:
            gt_df[col] = gt_df[col].astype(str).str.strip()

    # Merge
    merged_df = pd.merge(pred_df, gt_df, on=key_columns, how="inner")

    print(f"Matched rows for evaluation: {len(merged_df)}")

    if merged_df.empty:
        print("No matching rows found")
        return

    summary_results = []

    for pred_col in sentiment_prediction_cols:

        print("\n" + "=" * 80)
        print(f"Evaluating: {pred_col}")
        print("=" * 80)

        eval_df = merged_df[[ground_truth_col, pred_col]].dropna()

        y_true = eval_df[ground_truth_col]
        y_pred = eval_df[pred_col]

        labels = sorted(set(y_true.unique()) | set(y_pred.unique()))

        accuracy = accuracy_score(y_true, y_pred)

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)

        print(f"Samples evaluated: {len(eval_df)}")
        print(f"Accuracy          : {accuracy:.4f}")
        print(f"Macro Precision   : {precision_macro:.4f}")
        print(f"Macro Recall      : {recall_macro:.4f}")
        print(f"Macro F1          : {f1_macro:.4f}")
        print(f"Weighted Precision: {precision_weighted:.4f}")
        print(f"Weighted Recall   : {recall_weighted:.4f}")
        print(f"Weighted F1       : {f1_weighted:.4f}")
        print(f"Micro Precision   : {precision_micro:.4f}")
        print(f"Micro Recall      : {recall_micro:.4f}")
        print(f"Micro F1          : {f1_micro:.4f}\n")

        print("Classification Report:")
        print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])

        print("Confusion Matrix:")
        print(cm_df)

        summary_results.append({
        "model_column": pred_col,
        "samples_evaluated": len(eval_df),
        "accuracy": accuracy,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "weighted_precision": precision_weighted,
        "weighted_recall": recall_weighted,
        "weighted_f1": f1_weighted,
        "micro_precision": precision_micro,
        "micro_recall": recall_micro,
        "micro_f1": f1_micro
        })

    # Save summary
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv("data/classification_evaluation_summary.csv", index=False)

    print("\nSaved classification evaluation summary to: data/classification_evaluation_summary.csv")
    print(summary_df)


if __name__ == "__main__":
    main()