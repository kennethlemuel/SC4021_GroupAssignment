import pandas as pd
from transformers import pipeline
from senticnet.senticnet import SenticNet
from tqdm.auto import tqdm

from transformers.utils import logging
logging.set_verbosity_error()


def gronlp_subjectivity(df, text_col, batch_size: int = 16, confidence_threshold: float = 0.7):
    """
    Primary subjectivity classifier using GroNLP mDeBERTa model
    Labels: SUBJ -> Opinion, OBJ -> Fact
    Low confidence predictions (< confidence_threshold) are flagged for SenticNet fallback
    """
    classifier = pipeline("text-classification", model="GroNLP/mdebertav3-subjectivity-english")

    texts = df[text_col].fillna("").astype(str).tolist()
    labels = []
    confidences = []
    needs_fallback = []

    total = len(texts)

    for start_idx in tqdm(range(0, total, batch_size), desc="GroNLP subjectivity", unit="batch"):
        end_idx = min(start_idx + batch_size, total)
        batch_texts = [text.strip() for text in texts[start_idx:end_idx]]

        batch_labels = ["Subjective"] * len(batch_texts)
        batch_confidences = [0.0] * len(batch_texts)
        batch_fallback = [False] * len(batch_texts)

        non_empty_texts = []
        non_empty_positions = []

        for i, text in enumerate(batch_texts):
            if text:
                non_empty_texts.append(text)
                non_empty_positions.append(i)

        if non_empty_texts:
            results = classifier(non_empty_texts, batch_size=batch_size)

            for pos, result in zip(non_empty_positions, results):
                label = "Subjective" if result["label"] == "LABEL_1" else "Objective"
                confidence = round(float(result["score"]), 4)

                batch_labels[pos] = label
                batch_confidences[pos] = confidence

                # Flag low confidence predictions for SenticNet fallback
                if confidence < confidence_threshold:
                    batch_fallback[pos] = True

        labels.extend(batch_labels)
        confidences.extend(batch_confidences)
        needs_fallback.extend(batch_fallback)

    df["gronlp_subjectivity_label"] = labels
    df["gronlp_subjectivity_confidence"] = confidences
    df["needs_senticnet_fallback"] = needs_fallback

    return df


def senticnet_subjectivity(df, text_col):
    """
    SenticNet fallback for low-confidence (<0.7) GroNLP predictions
    Uses average absolute polarity magnitude across tokens as subjectivity signal
    High magnitude (>0.5) -> Subjective, else defer to original GroNLP label
    """
    sn = SenticNet()

    final_labels = []
    senticnet_scores = []
    senticnet_triggered = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="SenticNet fallback", unit="row"):
        gronlp_label = row["gronlp_subjectivity_label"]
        fallback_needed = row["needs_senticnet_fallback"]
        text = str(row[text_col]).lower()

        # For high confidence GroNLP predictions (>0.7), do not use SenticNet fallback
        if not fallback_needed:
            final_labels.append(gronlp_label)
            senticnet_scores.append(None)
            senticnet_triggered.append(False)
            continue

        # For low confidence GroNLP predictions (<0.7), use SenticNet
        words = text.split()
        polarity_scores = []

        for word in words:
            try:
                data = sn.concept(word)
                polarity = abs(float(data['polarity_value'])) # taking absolute of polarity value
                polarity_scores.append(polarity)
            except:
                pass  # word not in SenticNet

        if polarity_scores:
            avg_magnitude = sum(polarity_scores) / len(polarity_scores) # average polarity magnitude
        else:
            avg_magnitude = 0.0

        senticnet_scores.append(round(avg_magnitude, 4))

        if avg_magnitude > 0.3:
            # SenticNet has strong signal - override to Opinion
            final_labels.append("Subjective")
            senticnet_triggered.append(True)
        else:
            # SenticNet has no clear signal - fall back to GroNLP label
            final_labels.append(gronlp_label)
            senticnet_triggered.append(False)

    df["senticnet_triggered"] = senticnet_triggered
    df["senticnet_polarity_magnitude"] = senticnet_scores
    df["final_subjectivity_label"] = final_labels

    return df

def run_subjectivity_detection(input_csv, output_csv, text_col):

    df = pd.read_csv(input_csv)

    print("Running GroNLP subjectivity model...")
    df = gronlp_subjectivity(df, text_col)

    print("Running SenticNet fallback for low-confidence model predictions...")
    df = senticnet_subjectivity(df, text_col)

    # Summary
    total = len(df)
    fallback_count = df["needs_senticnet_fallback"].sum()
    triggered_count = df["senticnet_triggered"].sum()

    print(f"\n=== Summary ===")
    print(f"Total comments: {total}")
    print(f"High confidence GroNLP (no fallback): {total - fallback_count}")
    print(f"Low confidence flagged for SenticNet: {fallback_count}")
    print(f"SenticNet overrode GroNLP label: {triggered_count}")
    print(f"\nFinal label distribution:")
    print(df["final_subjectivity_label"].value_counts())

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\nSubjectivity detection completed")

    return df
