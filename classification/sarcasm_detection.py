'''
Sarcasm Detection of subjective comments using Hugging Face pipeline.
'''

import time
from pathlib import Path
import pandas as pd
from typing import Optional
from dataclasses import dataclass
import logging
from tqdm import tqdm
from transformers import pipeline

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()


# logging & progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
tqdm.pandas()


# project paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"


# file paths
INPUT_SUBJECTIVITY = DATA_DIR / "subjectivity_detection_results.csv"
OUTPUT_SARCASM = DATA_DIR / "sarcasm_detection_results.csv"


@dataclass
class SarcasmConfig:
    '''
    Config for sarcasm detection 
    '''
    sample_size: Optional[int] = None
    batch_size: int = 16
    model_name: str = "cardiffnlp/twitter-roberta-base-irony" 
    text_col: str = "cleaned_comments"
    subjectivity_col: str = "final_subjectivity_label"
    subjective_label: str = "Subjective"


class SarcasmDetector:
    '''
    Sarcasm detection on opinionated comments
    '''
    def __init__(self, config: SarcasmConfig = None):
        self.config = config or SarcasmConfig()
        logger.info(f"Initialized SarcasmDetector with config: {self.config}")

        self.classifier = pipeline(
            "text-classification",
            model=self.config.model_name
        )

    def map_label(self, raw_label):
        '''
        Map raw model labels to project labels
        '''
        raw_label = raw_label.lower()

        if raw_label in {"label_1", "irony"}:
            return "Sarcastic"
        elif raw_label in {"label_0", "non_irony"}:
            return "Not Sarcastic"

        return f"Unknown({raw_label})"

    def detect_sarcasm(self, df):
        '''
        Run sarcasm detection only on rows already marked as subjective
        '''
        df = df.copy()

        texts = df[self.config.text_col].fillna("").astype(str).tolist()
        subjectivity_labels = df[self.config.subjectivity_col].fillna("").astype(str).tolist()

        sarcasm_labels = []
        sarcasm_confidences = []
        sarcasm_applied = []

        total = len(texts)

        for start_index in tqdm(range(0, total, self.config.batch_size), desc="Sarcasm detection", unit="batch"):
            end_index = min(start_index + self.config.batch_size, total)

            batch_texts = [text.strip() for text in texts[start_index:end_index]]
            batch_subjectivity = subjectivity_labels[start_index:end_index]

            batch_labels = [None] * len(batch_texts)
            batch_confidences = [None] * len(batch_texts)
            batch_applied = [False] * len(batch_texts)

            valid_texts = []
            valid_positions = []

            for i, (text, subj) in enumerate(zip(batch_texts, batch_subjectivity)):
                if text and subj == self.config.subjective_label:
                    valid_texts.append(text)
                    valid_positions.append(i)

            if valid_texts:
                results = self.classifier(valid_texts, truncation=True, max_length=256)

                for pos, result in zip(valid_positions, results):
                    label = self.map_label(result["label"])
                    confidence = round(float(result["score"]), 4)

                    batch_labels[pos] = label
                    batch_confidences[pos] = confidence
                    batch_applied[pos] = True

            sarcasm_labels.extend(batch_labels)
            sarcasm_confidences.extend(batch_confidences)
            sarcasm_applied.extend(batch_applied)

            # logger.info(f"Sarcasm model processed {end_index}/{total}")

        df["sarcasm_label"] = sarcasm_labels
        df["sarcasm_confidence"] = sarcasm_confidences
        df["sarcasm_applied"] = sarcasm_applied

        return df


def run_sarcasm_detection(input_csv, output_csv, text_col):
    config = SarcasmConfig(sample_size=None, text_col=text_col)

    detector = SarcasmDetector(config)

    logger.info(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Sampling dataset for testing
    if config.sample_size:
        logger.info(f"Using sample of {config.sample_size} rows for testing...")
        df = df.head(config.sample_size)

    logger.info("Running sarcasm detection...")
    t0 = time.perf_counter()
    df_processed = detector.detect_sarcasm(df)
    elapsed_s = round(time.perf_counter() - t0, 4)

    subjective_count = (df_processed[config.subjectivity_col] == config.subjective_label).sum()
    sarcasm_count = (df_processed["sarcasm_label"] == "Sarcastic").sum()
    applied_count = df_processed["sarcasm_applied"].sum()

    logger.info("=== Summary ===")
    logger.info(f"Total comments: {len(df_processed)}")
    logger.info(f"Subjective comments: {subjective_count}")
    logger.info(f"Rows sarcasm model applied to: {applied_count}")
    logger.info(f"Sarcastic comments: {sarcasm_count}")
    logger.info(f"Sarcasm model elapsed time: {elapsed_s}s  ({round(applied_count / elapsed_s, 4) if elapsed_s > 0 else 'N/A'} records/s on subjective rows)")

    logger.info("Sarcasm label distribution:")
    logger.info(df_processed["sarcasm_label"].value_counts(dropna=False).to_string())

    try:
        logger.info(f"Saving processed data to {output_csv}...")
        df_processed.to_csv(output_csv, index=False, encoding="utf-8-sig")
    except PermissionError:
        logger.error(f"The file {output_csv} is currently open. Please close it and run the program again.")
        return
    except Exception as e:
        logger.error(f"An error occurred while saving the file: {e}")
        return

    logger.info(f"Sarcasm detection completed!")
    return df_processed, elapsed_s


if __name__ == "__main__":
    run_sarcasm_detection(input_csv=INPUT_SUBJECTIVITY, output_csv=OUTPUT_SARCASM, text_col="cleaned_comments")