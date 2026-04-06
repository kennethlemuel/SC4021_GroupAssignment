import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,Trainer,EarlyStoppingCallback,)

MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"
DATA_PATH = "classification/semeval_data/semeval_aspects_mapped.csv"
OUTPUT_DIR = "classification/checkpoints/deberta-base-finetuned"
BEST_DIR = "classification/checkpoints/deberta-base-finetuned-best"

MAX_LENGTH = 128 # max tokens. 80 is used by PyABSA but 128 is safer
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.00002
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

# Model's label scheme
ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}

# Run on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 1. Load and split data
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} samples") # Expected columns: text, aspect, label (0/1/2)
print(df["label"].value_counts().sort_index().rename({0:"negative", 1:"neutral", 2:"positive"}))

# Stratify on label to preserve class balance across splits
train_df, val_df = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df["label"])
print(f"\nTrain: {len(train_df)} | Val: {len(val_df)}")

# 2. Dataset class
class ABSADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.texts = dataframe["text"].tolist()
        self.aspects = dataframe["aspect"].tolist()
        self.labels = dataframe["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Text-pair input: (sentence, aspect), tokenizer produces: [CLS] sentence [SEP] aspect [SEP]
        encoding = self.tokenizer(
            self.texts[idx], # sentence
            self.aspects[idx], # aspect as second sequence
            max_length=self.max_length,
            padding="max_length",
            truncation=True, # truncates sentence if too long, keeps aspect intact
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# 3. Load tokenizer and model
print(f"\nLoading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,  # safe fallback if classifier head differs
)
model.to(device)

# 4. Build train and val datasets
train_dataset = ABSADataset(train_df, tokenizer, MAX_LENGTH)
val_dataset   = ABSADataset(val_df,   tokenizer, MAX_LENGTH)

# 5. Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4), "weighted_f1": round(weighted_f1, 4),}

# 6. Training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # Training schedule
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,

    # Optimiser
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO, # linear warmup for first 10% of steps

    # Evaluation and saving
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1", # use macro F1 since classes may be imbalanced
    greater_is_better=True,

    # Logging
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=50,
    report_to="none", # set to "wandb" if you want experiment tracking

    # Reproducibility
    seed=SEED
)

# 7. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # Stops training if macro_f1 doesn't improve for 2 consecutive epochs
)

print("\nStarting training...")
trainer.train()

# 9. Save best model
trainer.save_model(BEST_DIR)
tokenizer.save_pretrained(BEST_DIR)
print(f"\nBest model saved to: {BEST_DIR}")

# 10. Final evaluation with full classification report
print("\nFinal evaluation on validation set:")
predictions = trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

print(classification_report(labels, preds, target_names=["Negative", "Neutral", "Positive"]))

# 11. Per-aspect evaluation to see which aspects the model struggles with
print("\nPer-aspect macro F1:")
val_df = val_df.copy()
val_df["pred"] = preds

for aspect in sorted(val_df["aspect"].unique()):
    subset = val_df[val_df["aspect"] == aspect]
    if len(subset) < 5:
        print(f"  {aspect:15s}: too few samples ({len(subset)})")
        continue
    f1 = f1_score(subset["label"], subset["pred"], average="macro", zero_division=0)
    print(f"  {aspect:15s}: macro_f1={f1:.3f}  (n={len(subset)})")