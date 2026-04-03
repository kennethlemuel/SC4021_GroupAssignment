
import os
import warnings

from transformers import logging

from polarity2 import PolarityClassifier

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

def run_local_finetuned_example():
    print("\n=== LOCAL FINE-TUNED MODEL EXAMPLE ===")

    # Change this to your actual saved model folder
    local_model_path = "./local_finetuned"

    if not os.path.isdir(local_model_path):
        print(f"Local model folder not found: {local_model_path}")
        print("Update local_model_path to your saved fine-tuned model directory.")
        return

    clf = PolarityClassifier(
        mode="local_finetuned_with_senticnet",
        local_model_path=local_model_path,
    )

    texts = [
        "strong disagree it is easy to compare galaxy_s25 vs galaxy_s26 or even galaxy_s22 vs galaxy_s26 and say samsung is slacking but please let's blame samsung after analyzing the competition once iphone_17 120hz! finally! yay! but still uses usb 2__ __0 still charges at 25 watts still has a dual camera setup hardware wise that is still not par with the galaxy_s20 from 2020 which had 120hz usb 3__ __0 45 watt charging and a 3 camera setup with 8k ability! pixel_10 is far more competitive 120hz usb 3__ __0 faster 30w charging and the brand new 5x optical zoom lens this does bring some competition to samsung but the tenor g5 is still somewhat behind the snapdragon 8 gen 2 from galaxy_s23 series! so even this allows samsung to still be a strong contender without doing any real work so yeah complain all you want about the galaxy_s26 being boring but samsung has no reason to make any improvements when the competition is struggling to out do what they did in 2020 also do ask yourself why was the world singing praises for the iphone_17 if the samsung is underwhelming according to you? just the apple effect?",
        "to be honest i feel like they are going in a sort of app direction where apple focused more on optimizing thier battery rather than increasing it which is why its always neck and neck with samsung even though has a smaller battery it still goes with it wil life span if you want an upgrade go for another android phone at this point i herd thr transfer between android s is pretty good so theres no stoping you from an upgrade edit i was also going to mention its cheap and goes to a everyday use phone kind of branch rather than the showoff kind of branch that alot of phone brands a going like i phone huawei samsung and a couple more which is why i just feel like the galaxy_s26 is not really appealing and is missing alot of stuff and feels more like a phone that tries to tempt people in the upgrade into it",
        "the 1000 price for the galaxy_s26 with only 25w charging in the eu is honestly hard to justify i've been a long time samsung user i had the s10e please make smaller phones great again then the galaxy_s22 which i replaced very quickly because the battery life and the exynos chip were so disappointing right now i'm using the galaxy_s23 and i've had it for almost three years it still works just as well as it did on day one which shows how solid the device is however looking at the galaxy_s26 i simply don't see any real reason to upgrade as someone who has been a fan of samsung phones for years it's frustrating to feel like the company is no longer listening to its customers at this point it honestly feels like samsung has become even worse than apple in some aspects"
    ]
    

    for text in texts:
        print("\nInput Text:")
        print(text)
        print("\nOutput:")
        #print(clf.predict_single(text))
        print(clf.predict_single(text)['label'])


if __name__ == "__main__":
    run_local_finetuned_example()

"""
import os
import warnings
import pandas as pd

from transformers import logging
from polarity2 import PolarityClassifier

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.set_verbosity_error()


def normalize_label(label):
    if pd.isna(label):
        return None

    label = str(label).strip().lower()

    mapping = {
        "positive": "Positive",
        "pos": "Positive",
        "negative": "Negative",
        "neg": "Negative",
        "neutral": "Neutral",
        "neu": "Neutral",
    }

    return mapping.get(label, label.title())


def safe_read_csv(path):
    encodings_to_try = ["utf-8", "cp1252", "latin1"]

    for enc in encodings_to_try:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not read CSV file with tried encodings: {encodings_to_try}")


def build_classifier():
    local_model_path = "./local_finetuned"

    if not os.path.isdir(local_model_path):
        raise FileNotFoundError(
            f"Local model folder not found: {local_model_path}\n"
            f"Update local_model_path to your saved fine-tuned model directory."
        )

    clf = PolarityClassifier(
        mode="local_finetuned_with_senticnet",
        local_model_path=local_model_path,
    )
    return clf


def predict_label(clf, text):
    result = clf.predict_single(str(text))

    if isinstance(result, dict):
        label = result.get("label", None)
    else:
        label = result

    return normalize_label(label)


def run_local_finetuned_example():
    print("\n=== LOCAL FINE-TUNED MODEL EXAMPLE ===")

    clf = build_classifier()

    texts = [
        "strong disagree it is easy to compare galaxy_s25 vs galaxy_s26 or even galaxy_s22 vs galaxy_s26 and say samsung is slacking but please let's blame samsung after analyzing the competition once iphone_17 120hz! finally! yay! but still uses usb 2__ __0 still charges at 25 watts still has a dual camera setup hardware wise that is still not par with the galaxy_s20 from 2020 which had 120hz usb 3__ __0 45 watt charging and a 3 camera setup with 8k ability! pixel_10 is far more competitive 120hz usb 3__ __0 faster 30w charging and the brand new 5x optical zoom lens this does bring some competition to samsung but the tenor g5 is still somewhat behind the snapdragon 8 gen 2 from galaxy_s23 series! so even this allows samsung to still be a strong contender without doing any real work so yeah complain all you want about the galaxy_s26 being boring but samsung has no reason to make any improvements when the competition is struggling to out do what they did in 2020 also do ask yourself why was the world singing praises for the iphone_17 if the samsung is underwhelming according to you? just the apple effect?",
        "to be honest i feel like they are going in a sort of app direction where apple focused more on optimizing thier battery rather than increasing it which is why its always neck and neck with samsung even though has a smaller battery it still goes with it wil life span if you want an upgrade go for another android phone at this point i herd thr transfer between android s is pretty good so theres no stoping you from an upgrade edit i was also going to mention its cheap and goes to a everyday use phone kind of branch rather than the showoff kind of branch that alot of phone brands a going like i phone huawei samsung and a couple more which is why i just feel like the galaxy_s26 is not really appealing and is missing alot of stuff and feels more like a phone that tries to tempt people in the upgrade into it",
        "the 1000 price for the galaxy_s26 with only 25w charging in the eu is honestly hard to justify i've been a long time samsung user i had the s10e please make smaller phones great again then the galaxy_s22 which i replaced very quickly because the battery life and the exynos chip were so disappointing right now i'm using the galaxy_s23 and i've had it for almost three years it still works just as well as it did on day one which shows how solid the device is however looking at the galaxy_s26 i simply don't see any real reason to upgrade as someone who has been a fan of samsung phones for years it's frustrating to feel like the company is no longer listening to its customers at this point it honestly feels like samsung has become even worse than apple in some aspects"
    ]

    for text in texts:
        print("\nInput Text:")
        print(text)
        print("\nOutput:")
        print(predict_label(clf, text))


def run_csv_check(
    csv_path="annotation_candidates.csv",
    text_column="text",
    annotator_column="annotator_1",
    max_rows=600,
    output_path="annotation_candidates_comparison_600.csv",
):
    print("\n=== CSV CHECK MODE ===")

    clf = build_classifier()
    df = safe_read_csv(csv_path)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in {csv_path}")

    if annotator_column not in df.columns:
        raise ValueError(f"Column '{annotator_column}' not found in {csv_path}")

    df = df.head(max_rows).copy()

    predictions = []
    gold_labels = []
    matches = []

    for _, row in df.iterrows():
        text = row[text_column]
        gold = normalize_label(row[annotator_column])

        pred = predict_label(clf, text)

        predictions.append(pred)
        gold_labels.append(gold)
        matches.append(pred == gold if gold is not None else False)

    df["predicted_label"] = predictions
    df["gold_label"] = gold_labels
    df["match"] = matches

    # Evaluate only on non-neutral ground-truth labels.
    valid_df = df[df["gold_label"].isin(["Positive", "Negative"])].copy()

    if len(valid_df) > 0:
        accuracy = valid_df["match"].mean()
        print(f"\nCompared rows: {len(valid_df)}")
        print(f"Accuracy: {accuracy:.4f}")

        confusion = pd.crosstab(
            valid_df["gold_label"],
            valid_df["predicted_label"],
            rownames=["Annotator"],
            colnames=["Model"],
            dropna=False,
        )

        print("\nConfusion Matrix:")
        print(confusion)
    else:
        print("\nNo valid annotated rows found for comparison.")
        confusion = pd.DataFrame()

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved detailed results to: {output_path}")

    if not confusion.empty:
        confusion_output = output_path.replace(".csv", "_confusion_matrix.csv")
        confusion.to_csv(confusion_output, encoding="utf-8-sig")
        print(f"Saved confusion matrix to: {confusion_output}")


if __name__ == "__main__":
    # Example mode:
    # run_local_finetuned_example()

    # CSV import + first 600 row comparison mode:
    run_csv_check(
        csv_path="annotation_candidates.csv",
        text_column="text",
        annotator_column="annotator_1 (ken)",
        max_rows=600,
        output_path="annotation_candidates_comparison_600.csv",
    )
    """