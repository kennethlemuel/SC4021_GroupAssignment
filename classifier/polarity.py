from __future__ import annotations

from senticnet.senticnet import SenticNet

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import re

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
            "rule_based": self.predict_rule_based(text), # NEW
        }

        vote_scores = {"positive": 0, "negative": 0}

        weights = {
            #"local_finetuned": 4,
            #"senticnet": 1,
            "local_finetuned": 3,
            "senticnet": 2,
            "rule_based": 2,
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

    '''
    def predict_rule_based(self, text: str) -> str:
        text_lower = text.lower()

        # Rule 1: strong negative phrases
        if any(phrase in text_lower for phrase in [
            "never buy", "worst", "terrible", "awful", "hate", "disappointing"
        ]):
            return "negative"

        # Rule 2: strong positive phrases
        if any(phrase in text_lower for phrase in [
            "love", "amazing", "excellent", "perfect", "best"
        ]):
            return "positive"

        # Rule 3: negation handling
        if "not good" in text_lower or "not great" in text_lower:
            return "negative"

        if "not bad" in text_lower:
            return "positive"

        # Rule 4: contrast handling ("but")
        if "but" in text_lower:
            parts = text_lower.split("but")
            if len(parts) > 1:
                return self.predict_rule_based(parts[-1])  # focus on last clause

        # Rule 5: fallback → neutral
        return "neutral"
    '''

    def predict_rule_based(self, text: str) -> str:
        text_lower = text.lower().strip()

        # -------------------------
        # 0. Neutral / informational patterns
        # -------------------------
        
        neutral_starts = (
            "would you recommend",
            "should i",
            "can i",
            "does anyone know",
            "what is",
            "how is",
            "if you",
            "for context",
        )
        

        if text_lower.endswith("?") or any(text_lower.startswith(p) for p in neutral_starts):
            return "neutral"
            #return "positive"

        # Mostly factual/spec-like language with little sentiment
        factual_markers = [
            "uses usb", "charges at", "mah", "hz", "watts", "4k", "8k",
            "released", "supports", "cap at", "sensor", "adapter",
        ]
        if sum(marker in text_lower for marker in factual_markers) >= 2:
            return "neutral"
            #return "positive"

        # -------------------------
        # 1. Strong phrase lists
        # -------------------------
        strong_positive_phrases = [
            "huge jump", "very good jump", "worth it", "totally worth it",
            "better in most ways", "could not be happier", "very good thing",
            "really good", "so good", "just perfect", "my favorite",
            "best camera", "best phone", "strong contender", "happy with",
            "gladly traded", "love it", "enjoying my", "great upgrade",
            "solid decisions", "refreshing to see", "good opportunity",
        ]

        strong_negative_phrases = [
            "no reason to upgrade", "not worth it", "can't justify",
            "cannot justify", "could have used", "same phone",
            "pile of trash", "trash hardware", "actual trash",
            "worse than", "looks worse", "video performance is so bad",
            "thoroughly disappointed", "returned it", "giving me a pause",
            "nothing good has come out", "boring", "slacking",
            "underwhelming", "iphone clones", "feels like just",
            "not up to it", "struggling to", "doesn't happen with",
        ]

        strong_positive_words = {
            "love", "great", "amazing", "excellent", "perfect", "best",
            "better", "improved", "good", "happy", "solid", "smooth",
            "favorite", "enjoying", "refreshing", "strong", "worth",
            "upgrade", "recommend",
        }

        strong_negative_words = {
            "worst", "terrible", "awful", "hate", "disappointing",
            "disappointed", "bad", "worse", "boring", "slacking",
            "trash", "overpriced", "laggy", "grainy", "broken",
            "annoying", "underwhelming", "pause", "clone", "clones",
            "problem", "issues",
        }

        intensifiers = {
            "very": 1, "really": 1, "so": 1, "much": 1,
            "far": 1, "definitely": 1, "absolutely": 1,
        }

        downtoners = {"slightly", "maybe", "somewhat", "kinda", "kind of", "a little"}

        negations = {
            "not", "no", "never", "can't", "cannot", "dont", "don't",
            "isn't", "wasn't", "won't", "shouldn't", "couldn't",
        }

        # -------------------------
        # 2. Split on contrast markers
        #    Later clause gets more weight
        # -------------------------
        clauses = re.split(r"\b(?:but|however|though|although|except)\b", text_lower)
        contrast_present = len(clauses) > 1

        total_score = 0.0

        for i, clause in enumerate(clauses):
            words = re.findall(r"[a-z0-9']+", clause)
            if not words:
                continue

            clause_score = 0.0

            # Later clauses after "but/however" matter more
            clause_weight = 1.0
            if contrast_present and i == len(clauses) - 1:
                clause_weight = 1.5

            # Phrase hits first
            for phrase in strong_positive_phrases:
                if phrase in clause:
                    clause_score += 3

            for phrase in strong_negative_phrases:
                if phrase in clause:
                    clause_score -= 3

            # Word-level scoring with negation/intensifier handling
            for j, word in enumerate(words):
                prev = words[j - 1] if j - 1 >= 0 else ""
                prev2 = words[j - 2] if j - 2 >= 0 else ""

                is_negated = prev in negations or prev2 in negations
                boost = 1.0

                if prev in intensifiers or prev2 in intensifiers:
                    boost += 0.5
                if prev in downtoners or prev2 in downtoners:
                    boost -= 0.3

                # domain-specific upgrade / downgrade cues
                if word == "upgrade":
                    if any(p in clause for p in ["no reason to upgrade", "not worth upgrading", "can't justify upgrading"]):
                        clause_score -= 2 * boost
                    else:
                        clause_score += 1 * boost

                #if word in {"better", "improved", "worth", "good", "great", "love", "perfect", "best", "happy"}:
                if word in strong_positive_words:
                    delta = 1.5 * boost
                    clause_score += (-delta if is_negated else delta)

                #elif word in {"worse", "bad", "terrible", "awful", "trash", "boring", "disappointing", "underwhelming", "hate"}:
                elif word in strong_negative_words:
                    delta = 1.5 * boost
                    clause_score += (delta if is_negated else -delta)

            # comparison cues
            if "better than" in clause:
                clause_score += 2
            if "worse than" in clause:
                clause_score -= 2
            if "same as" in clause or "basically the same" in clause:
                clause_score -= 1.5
            if "no reason to" in clause:
                clause_score -= 2
            if "worth it" in clause:
                clause_score += 2
            if "not worth it" in clause:
                clause_score -= 2.5

            total_score += clause_score * clause_weight

        # -------------------------
        # 3. Final label
        # -------------------------
        if total_score >= 1.5:
            return "positive"
        if total_score <= -1.5:
            return "negative"
        return "neutral"

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
    '''
    '''
    def predict_single(self, text: str) -> Dict[str, Any]:
        if self.mode == "roberta_only":
            result = self.predict_roberta_only(text)
            return result.to_dict()
        elif self.mode == "local_finetuned":
            result = self.predict_local_finetuned_only(text)
            return result.to_dict()
        elif self.mode == "local_finetuned_with_senticnet":
            # short text -> normal hybrid
            if len(text.split()) <= 40:
                result = self.predict_local_finetuned_with_senticnet(text)
                return result.to_dict()
            # long text -> sentence-level aggregation
            return self.predict_document_hybrid(text)
        else:
            result = self.predict_weighted_ensemble_no_bert(text)
            return result.to_dict()
    '''
    def predict_single(self, text: str) -> Dict[str, Any]:
        if self.mode == "roberta_only":
            result = self.predict_roberta_only(text)
            return result.to_dict()

        elif self.mode == "local_finetuned":
            result = self.predict_local_finetuned_only(text)
            return result.to_dict()

        elif self.mode == "local_finetuned_with_senticnet":
            # long review -> sentence-weighted aggregation
            if len(text.split()) > 40:
                return self.predict_document_hybrid(text)

            # short comment -> regular hybrid
            result = self.predict_local_finetuned_with_senticnet(text)
            return result.to_dict()

        else:
            result = self.predict_weighted_ensemble_no_bert(text)
            return result.to_dict()

    '''
    def predict_document_hybrid(self, text: str) -> Dict[str, Any]:
        segments = self.split_into_sentences(text)

        if not segments:
            return self.predict_single(text)

        segment_results = []
        positive_score = 0.0
        negative_score = 0.0

        n = len(segments)

        for i, seg in enumerate(segments):
            result = self.predict_local_finetuned_with_senticnet(seg).to_dict()
            segment_results.append(result)

            label = result["label"]
            votes = result["vote_scores"]
            margin = abs(votes["positive"] - votes["negative"])
            strength = max(1, margin)

            # later segments get slightly more weight
            # position_weight = 1.0 + (0.3 * i / max(1, n - 1)) # linear style
            alpha = 1.4
            position_weight = alpha ** (i / max(1, n - 1) * 4) # exponential
            score = strength * position_weight

            if label == "positive":
                positive_score += score
            elif label == "negative":
                negative_score += score

        final_label = "positive" if positive_score >= negative_score else "negative"

        return {
            "text": text,
            "label": final_label,
            "segment_results": segment_results,
            "document_vote_scores": {
                "positive": round(positive_score, 3),
                "negative": round(negative_score, 3),
            },
            "final_method": "sentence_level_hybrid_aggregation",
        }
    '''

    def predict_document_hybrid(self, text: str) -> Dict[str, Any]:
        segments = self.split_into_segments(text)

        if not segments:
            return self.predict_segment_votes(text)

        segment_results = []
        document_vote_scores = {"positive": 0.0, "negative": 0.0}

        n = len(segments)

        for i, seg in enumerate(segments):
            seg_result = self.predict_segment_votes(seg)
            segment_results.append(seg_result)

            seg_votes = seg_result["vote_scores"]

            # Strong end weighting:
            # first segment ~1.0x
            # last segment ~4.0x
            if n == 1:
                position_weight = 1.0
            else:
                position_weight = 1.0 + 3.0 * (i / (n - 1))

            document_vote_scores["positive"] += seg_votes["positive"] * position_weight
            document_vote_scores["negative"] += seg_votes["negative"] * position_weight

        diff = document_vote_scores["positive"] - document_vote_scores["negative"]
        total = document_vote_scores["positive"] + document_vote_scores["negative"]

        last_segment = segment_results[-1]
        last_label = last_segment["label"]
        last_votes = last_segment["vote_scores"]

        last_strength = abs(last_votes["positive"] - last_votes["negative"])

        final_label = (
            "positive"
            if document_vote_scores["positive"] >= document_vote_scores["negative"]
            else "negative"
        )

        # print(last_label, last_strength, diff, total, final_label)

        # If final sentence is strong and overall difference is small → override
        if last_label == "negative":
            if diff > 0:  # currently positive
                if diff < 0.25 * total: #and last_strength >= 2:
                    final_label = "negative"

        elif last_label == "positive":
            if diff < 0:  # currently negative
                if abs(diff) < 0.25 * total: #and last_strength >= 2:
                    final_label = "positive"

        return {
            "text": text,
            "label": final_label,
            "segment_results": segment_results,
            "vote_scores": {
                "positive": round(document_vote_scores["positive"], 3),
                "negative": round(document_vote_scores["negative"], 3),
            },
            "final_method": "sentence_weighted_hybrid",
        }

    def split_into_sentences(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        # split on punctuation first
        parts = re.split(r"(?<=[.!?])\s+", text)

        # fallback for long run-on text with no punctuation:
        # split on discourse markers often used in reviews
        final_parts = []
        for part in parts:
            subparts = re.split(
                r"\b(?:but|however|though|although|also|edit|because|which is why)\b",
                part,
                flags=re.IGNORECASE,
            )
            final_parts.extend([p.strip(" ,;:-") for p in subparts if p.strip(" ,;:-")])

        return [p for p in final_parts if p]
    
    def split_into_segments(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        # first split on sentence punctuation
        segments = re.split(r"(?<=[.!?])\s+", text)

        cleaned = []
        for seg in segments:
            seg = seg.strip()
            if seg:
                cleaned.append(seg)

        # fallback: if the whole review is one giant run-on sentence,
        # split on common discourse markers too
        if len(cleaned) <= 1:
            parts = re.split(
                r"\b(?:but|however|though|although|except|also|edit|because|which is why)\b",
                text,
                flags=re.IGNORECASE,
            )
            cleaned = [p.strip(" ,;:-") for p in parts if p.strip(" ,;:-")]

        return cleaned
    
    def predict_segment_votes(self, text: str) -> Dict[str, Any]:
        model_predictions = {
            "local_finetuned": self.predict_local_finetuned(text),
            "senticnet": self.predict_senticnet(text),
            "rule_based": self.predict_rule_based(text),
        }

        base_weights = {
            "local_finetuned": 3,
            "senticnet": 2,
            "rule_based": 2,
        }

        vote_scores = {"positive": 0.0, "negative": 0.0}

        for model_name, pred in model_predictions.items():
            if pred in ("positive", "negative"):
                vote_scores[pred] += base_weights[model_name]

        label = "positive" if vote_scores["positive"] >= vote_scores["negative"] else "negative"

        return {
            "text": text,
            "label": label,
            "model_predictions": model_predictions,
            "vote_scores": vote_scores,
        }
        
    ######

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