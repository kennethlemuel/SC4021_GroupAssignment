import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

FIXED_ASPECTS_KEYWORDS = {
    "camera": ["camera", "photo", "photos", "video", "videos", "zoom", "selfie", "lens", "image quality"],
    "battery": ["battery", "battery life", "drain", "screen on time", "endurance", "overheat", "thermals", "heat"],
    "display": ["display", "screen", "panel", "brightness", "refresh rate", "120hz", "60hz", "ltpo", "resolution", "bezel"],
    "price": ["price", "cost", "value", "affordable", "expensive", "pricing", "worth", "budget", "price tag", "discount", "deal"],
    "software": ["software", "operating system", "os", "app", "application", "program", "windows", "driver", "interface", "firmware", "bloatware", "update", "gui", "bugs"],
    "design": ["design", "build", "keyboard", "trackpad", "touchpad", "weight", "port", "hinge", "body", "casing", "construction", "material", "looks", "size", "small", "compact", "weight", "titanium", "aluminium", "corner"],
    "performance": ["performance", "speed", "processor", "cpu", "chip", "benchmark", "lag", "boot", "startup", "processing power", "cores"],
    "charging": ["charging", "charger", "charge", "power adapter", "plug", "cable", "watt", "fast charge", "power supply", "cord"],
    "storage": ["storage", "hard drive", "ssd", "disk", "capacity", "memory", "ram", "gb", "tb", "hdd"],
}

# 1. Load the dataset
df = pd.read_csv("classification/semeval_data/SemEval2014_Task4_Laptop_Train_v2.csv")
df.columns = ["id", "text", "aspect_term", "polarity", "from", "to"]
print("Total rows:", len(df))

# 2. Basic cleaning
df = df[df["polarity"] != "conflict"].copy()
print(f"After removing 'conflict' label: {len(df)} rows")
df["aspect_term"] = df["aspect_term"].str.lower().str.strip()

# 3. Get unique aspect terms
unique_aspects = df["aspect_term"].unique().tolist()
print(f"Total unique aspects in SemEval dataset: {len(unique_aspects)}")

# 4. Embed keywords for our 9 fixed aspects and SemEval aspects
print("\nLoading sentence transformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Embed each aspect's keyword list individually (dict: aspect -> (num_keywords, 384))
aspect_keyword_embeddings = {}
for aspect, keywords in FIXED_ASPECTS_KEYWORDS.items():
    aspect_keyword_embeddings[aspect] = embedder.encode(keywords, convert_to_tensor=True)

fixed_aspects_list = list(FIXED_ASPECTS_KEYWORDS.keys())

# Embed all unique SemEval aspect terms (num_unique_aspects, 384)
semeval_embeddings = embedder.encode(unique_aspects, convert_to_tensor=True)
print("Embeddings done")

# 5. Compute max similarity and build mapping
THRESHOLD = 0.5 # (0.4 to 0.5 seems good)

mapping = {} # semeval aspect term -> fixed aspect (or None if discarded)
mapping_scores = {} # stores best score per semeval term for inspection

for i, semeval_asp in enumerate(unique_aspects):
    semeval_emb = semeval_embeddings[i]

    best_aspect = None
    best_score = -1

    for aspect, kw_embeddings in aspect_keyword_embeddings.items():
        # Similarity between this SemEval term and all keywords of this aspect
        sims = util.cos_sim(semeval_emb, kw_embeddings)[0]

        # Take the max-> one strong keyword match is enough to assign
        max_sim = sims.max().item()

        if max_sim > best_score:
            best_score = max_sim
            best_aspect = aspect

    mapping[semeval_asp] = best_aspect if best_score >= THRESHOLD else None
    mapping_scores[semeval_asp] = round(best_score, 3)

# 6. Inspect mapping results
print("\n" + "="*60)
print("MAPPING RESULTS WITH SCORES")
print("="*60)

grouped = defaultdict(list)
for semeval_asp, fixed_asp in mapping.items():
    grouped[fixed_asp].append((semeval_asp, mapping_scores[semeval_asp]))

for asp in fixed_aspects_list:
    print(f"\n=== {asp} ===")
    for term, score in sorted(grouped[asp], key=lambda x: -x[1]):
        print(f"  {score:.3f}  {term}")

print(f"\n=== DISCARDED ===")
for term, score in sorted(grouped[None], key=lambda x: -x[1]):
    print(f"  {score:.3f}  {term}")

# 7. Apply mapping to build final dataset
label2id = {"negative": 0, "neutral": 1, "positive": 2}

final_rows = []
for _, row in df.iterrows():
    fixed_asp = mapping.get(row["aspect_term"])
    if fixed_asp is None:
        continue
    final_rows.append({
        "text":   row["text"],
        "aspect": fixed_asp,
        "label":  label2id[row["polarity"]]
    })

final_df = pd.DataFrame(final_rows)
print(f"\nKept {len(final_df)}/{len(df)} samples after mapping")

# 8. Check class distribution per aspect
id2label = {0: "negative", 1: "neutral", 2: "positive"}

print("\n" + "="*60)
print("SAMPLE COUNTS PER ASPECT")
print("="*60)
for asp in fixed_aspects_list:
    subset = final_df[final_df["aspect"] == asp]
    counts = subset["label"].value_counts().to_dict()
    neg = counts.get(0, 0)
    neu = counts.get(1, 0)
    pos = counts.get(2, 0)
    print(f"{asp:15s}: total={len(subset):4d} | neg={neg:3d} | neu={neu:3d} | pos={pos:3d}")

# 9. Save final mapped dataset
output_path = "classification/semeval_data/semeval_aspects_mapped.csv"
final_df.to_csv(output_path, index=False)
print(f"\nSaved {len(final_df)} samples to {output_path}")
