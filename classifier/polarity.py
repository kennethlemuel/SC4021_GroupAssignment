from __future__ import annotations

from senticnet.senticnet import SenticNet

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import os
import warnings
import inspect
import torch

from transformers import (
    pipeline,
    logging,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Suppress noisy logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.set_verbosity_error()


@dataclass
class PolarityResult:
    text: str
    label: str
    model_predictions: Dict[str, str]
    vote_scores: Dict[str, int]
    final_method: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "model_predictions": self.model_predictions,
            "vote_scores": self.vote_scores,
            "final_method": self.final_method,
        }


class PolarityClassifier:
    VALID_MODES = {"roberta_only", "weighted_ensemble_no_bert", "local_finetuned", "local_finetuned_with_senticnet"}

    def __init__(
        self,
        mode: str = "roberta_only",
        use_cpu: bool = True,
        vader_threshold: float = 0.05,
        weights: Optional[Dict[str, int]] = None,
        local_model_path: Optional[str] = None,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported mode: {mode}. Valid modes: {sorted(self.VALID_MODES)}"
            )

        self.mode = mode
        self.vader_threshold = vader_threshold
        self.device = -1 if use_cpu else 0
        self.local_model_path = local_model_path
        self._senticnet = None

        self.weights = weights or {
            "roberta": 3,
            "distilbert": 2,
            "vader": 1,
            "senticnet": 1,
        }

        self._roberta_clf = None
        self._distilbert_clf = None
        self._vader = None
        self._local_finetuned_tokenizer = None
        self._local_finetuned_model = None

        self.positive_words = {
            "love", "great", "amazing", "best", "smooth", "premium",
            "helpful", "fast", "stunning", "flawless", "recommend",
            "phenomenal", "vibrant", "durable", "clear", "excellent",
            "fantastic", "awesome", "perfect", "good", "stable",
        }

        self.negative_words = {
            "terrible", "worst", "overheats", "disappointing", "slow",
            "grainy", "broken", "crash", "rude", "unhelpful",
            "overpriced", "freezing", "muffled", "distorted", "laggy",
            "bad", "awful", "poor", "useless", "annoying",
        }

    def _get_senticnet(self):
        if self._senticnet is None:
            self._senticnet = SenticNet()
        return self._senticnet

    def _get_roberta(self):
        if self._roberta_clf is None:
            self._roberta_clf = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=self.device,
            )
        return self._roberta_clf

    def _get_distilbert(self):
        if self._distilbert_clf is None:
            self._distilbert_clf = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=self.device,
            )
        return self._distilbert_clf

    def _get_vader(self):
        if self._vader is None:
            self._vader = SentimentIntensityAnalyzer()
        return self._vader

    def _get_local_finetuned(self):
        if self.local_model_path is None:
            raise ValueError(
                "local_model_path must be provided when mode='local_finetuned'"
            )

        if self._local_finetuned_tokenizer is None:
            self._local_finetuned_tokenizer = AutoTokenizer.from_pretrained(
                self.local_model_path
            )

        if self._local_finetuned_model is None:
            self._local_finetuned_model = AutoModelForSequenceClassification.from_pretrained(
                self.local_model_path
            )
            self._local_finetuned_model.to("cpu" if self.device == -1 else "cuda")
            self._local_finetuned_model.eval()

        return self._local_finetuned_tokenizer, self._local_finetuned_model

    def predict_distilbert(self, text: str) -> str:
        clf = self._get_distilbert()
        result = clf(text, truncation=True, max_length=128)[0]
        label = result["label"].lower()
        return "positive" if "pos" in label else "negative"

    def predict_roberta(self, text: str) -> str:
        clf = self._get_roberta()
        result = clf(text, truncation=True, max_length=128)[0]
        label = result["label"].lower()

        if "positive" in label:
            return "positive"
        if "negative" in label:
            return "negative"
        if "neutral" in label:
            return "neutral"

        return "negative"

    def predict_vader(self, text: str) -> str:
        vader = self._get_vader()
        score = vader.polarity_scores(text)["compound"]
        return "positive" if score >= self.vader_threshold else "negative"
    '''
    def predict_senticnet(self, text: str) -> str:
        words = text.lower().split()
        score = 0

        for word in words:
            cleaned = word.strip(".,!?;:'\"()[]{}")
            if cleaned in self.positive_words:
                score += 1
            elif cleaned in self.negative_words:
                score -= 1

        return "positive" if score >= 0 else "negative"
    '''

    def predict_senticnet(self, text: str) -> str:
        sn = self._get_senticnet()

        # naive concept extraction: unigrams + bigrams
        tokens = [t.strip(".,!?;:'\"()[]{}").lower() for t in text.split()]
        tokens = [t for t in tokens if t]

        concepts = set(tokens)
        concepts.update(
            f"{tokens[i]}_{tokens[i+1]}"
            for i in range(len(tokens) - 1)
        )

        scores = []

        for concept in concepts:
            try:
                score = float(sn.polarity_value(concept))
                scores.append(score)
            except Exception:
                continue

        if not scores:
            return "neutral"

        avg_score = sum(scores) / len(scores)

        if avg_score >= 0:
            return "positive"
        if avg_score < 0:
            return "negative"
        # return "neutral"

    def predict_local_finetuned(self, text: str) -> str:
        tokenizer, model = self._get_local_finetuned()

        encoded = tokenizer(
            text,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt",
        )

        device = next(model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        valid_keys = set(inspect.signature(model.forward).parameters.keys())
        encoded = {k: v for k, v in encoded.items() if k in valid_keys}

        with torch.no_grad():
            outputs = model(**encoded)
            pred_id = int(torch.argmax(outputs.logits, dim=-1).item())

        id2label = getattr(model.config, "id2label", None)
        if isinstance(id2label, dict) and pred_id in id2label:
            label = str(id2label[pred_id]).lower()
        elif isinstance(id2label, dict) and str(pred_id) in id2label:
            label = str(id2label[str(pred_id)]).lower()
        else:
            label = f"label_{pred_id}"

        if "positive" in label:
            return "positive"
        if "negative" in label:
            return "negative"
        if "neutral" in label:
            return "neutral"

        return label

    def predict_local_finetuned_with_senticnet(self, text: str) -> PolarityResult:
        model_predictions = {
            "local_finetuned": self.predict_local_finetuned(text),
            "senticnet": self.predict_senticnet(text),
        }

        vote_scores = {"positive": 0, "negative": 0}

        weights = {
            "local_finetuned": 4,
            "senticnet": 1,
        }

        for model_name, pred in model_predictions.items():
            if pred in ("positive", "negative"):
                vote_scores[pred] += weights[model_name]

        final_label = (
            "positive"
            if vote_scores["positive"] >= vote_scores["negative"]
            else "negative"
        )

        return PolarityResult(
            text=text,
            label=final_label,
            model_predictions=model_predictions,
            vote_scores=vote_scores,
            final_method="local_finetuned_with_senticnet",
        )

    def predict_roberta_only(self, text: str) -> PolarityResult:
        label = self.predict_roberta(text)
        return PolarityResult(
            text=text,
            label=label,
            model_predictions={"roberta": label},
            vote_scores={
                "positive": int(label == "positive"),
                "negative": int(label == "negative"),
            },
            final_method="roberta_only",
        )

    def predict_local_finetuned_only(self, text: str) -> PolarityResult:
        label = self.predict_local_finetuned(text)
        return PolarityResult(
            text=text,
            label=label,
            model_predictions={"local_finetuned": label},
            vote_scores={
                "positive": int(label == "positive"),
                "negative": int(label == "negative"),
            },
            final_method="local_finetuned",
        )

    def predict_weighted_ensemble_no_bert(self, text: str) -> PolarityResult:
        model_predictions = {
            "roberta": self.predict_roberta(text),
            "distilbert": self.predict_distilbert(text),
            "vader": self.predict_vader(text),
            "senticnet": self.predict_senticnet(text),
        }

        vote_scores = {"positive": 0, "negative": 0}

        for model_name, pred in model_predictions.items():
            if pred in ("positive", "negative"):
                vote_scores[pred] += self.weights[model_name]

        final_label = (
            "positive"
            if vote_scores["positive"] >= vote_scores["negative"]
            else "negative"
        )

        return PolarityResult(
            text=text,
            label=final_label,
            model_predictions=model_predictions,
            vote_scores=vote_scores,
            final_method="weighted_ensemble_no_bert",
        )
    '''
    def predict_single(self, text: str) -> Dict[str, Any]:
        if self.mode == "roberta_only":
            result = self.predict_roberta_only(text)
        elif self.mode == "local_finetuned":
            result = self.predict_local_finetuned_only(text)
        else:
            result = self.predict_weighted_ensemble_no_bert(text)

        return result.to_dict()
    '''
    def predict_single(self, text: str) -> Dict[str, Any]:
        if self.mode == "roberta_only":
            result = self.predict_roberta_only(text)
        elif self.mode == "local_finetuned":
            result = self.predict_local_finetuned_only(text)
        elif self.mode == "local_finetuned_with_senticnet":
            result = self.predict_local_finetuned_with_senticnet(text)
        else:
            result = self.predict_weighted_ensemble_no_bert(text)

        return result.to_dict()

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.predict_single(text) for text in texts]

    def predict_records(
        self,
        records: List[Dict[str, Any]],
        text_key: str = "text",
        subjectivity_key: str = "subjectivity",
        opinion_value: str = "opinion",
    ) -> List[Dict[str, Any]]:
        output_records: List[Dict[str, Any]] = []

        for record in records:
            new_record = dict(record)

            text = str(new_record.get(text_key, "")).strip()
            subjectivity = new_record.get(subjectivity_key)

            if subjectivity == opinion_value and text:
                result = self.predict_single(text)
                new_record["polarity"] = result["label"]
                new_record["polarity_meta"] = {
                    "final_method": result["final_method"],
                    "vote_scores": result["vote_scores"],
                    "model_predictions": result["model_predictions"],
                }
            else:
                new_record["polarity"] = None
                new_record["polarity_meta"] = None

            output_records.append(new_record)

        return output_records