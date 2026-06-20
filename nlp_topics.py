"""
nlp_topics.py
─────────────────────────────────────────────────────────────────────
Unsupervised topic modeling on PSE healthcare incident descriptions
using BERTopic with sentence-transformer embeddings.

Discovers recurring themes across ~4,230 incident narratives WITHOUT
predefined keywords — then tracks how those themes evolved over the
conflict (topics-over-time), directly extending the conflict-phase work.

Run AFTER feature2.py.  Reads pse_healthcare_featured_v2.csv.

Outputs:
  pse_topics.csv          each incident + its assigned topic
  topic_summary.csv       topic id, size, representative words
  topics_over_time.html   interactive theme-evolution chart

Install once:
  pip install bertopic sentence-transformers --break-system-packages
─────────────────────────────────────────────────────────────────────
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

BASE = "/Users/andreisales/Desktop/document idf"
DATA = f"{BASE}/pse_healthcare_featured_v2.csv"

# ── Load & filter ─────────────────────────────────────────────────────────────
print("Loading...")
df = pd.read_csv(DATA, parse_dates=["Date"])
df = df[df["description_clean"].notna()].copy()
df = df[df["description_clean"].str.len() > 25].copy()   # drop stubs
docs = df["description_clean"].astype(str).tolist()
print(f"  Documents for modeling: {len(docs):,}")

# ── Domain stopwords ──────────────────────────────────────────────────────────
# These terms appear in nearly every record; left in, they'd dominate every
# topic and wash out the actual distinguishing themes.
domain_stops = [
    "medical", "facility", "facilities", "health", "healthcare", "worker",
    "workers", "hospital", "clinic", "israeli", "forces", "force",
    "palestinian", "gaza", "west", "bank", "reported", "report",
    "attack", "attacked", "incident", "staff", "care", "said", "according",
]
stops = list(ENGLISH_STOP_WORDS) + domain_stops

vectorizer = CountVectorizer(
    stop_words=stops,
    ngram_range=(1, 2),     # unigrams + bigrams ("ambulance crew", "denied access")
    min_df=5,               # term must appear in >=5 docs
)

# ── Embeddings + model ────────────────────────────────────────────────────────
print("Embedding (first run downloads ~80MB model)...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

topic_model = BERTopic(
    embedding_model=embed_model,
    vectorizer_model=vectorizer,
    min_topic_size=25,      # a theme needs >=25 incidents to count
    nr_topics="auto",       # merge similar topics automatically
    calculate_probabilities=False,
    verbose=True,
)

print("Fitting topics...")
topics, _ = topic_model.fit_transform(docs)
df["topic"] = topics

# Reassign outliers (topic -1) to their nearest real topic for cleaner output
print("Reducing outliers...")
df["topic_reduced"] = topic_model.reduce_outliers(docs, topics)

# ── Topic summary ─────────────────────────────────────────────────────────────
info = topic_model.get_topic_info()
print("\nDiscovered topics:")
print(info[["Topic", "Count", "Name"]].head(20).to_string(index=False))

info.to_csv(f"{BASE}/topic_summary.csv", index=False)
df.to_csv(f"{BASE}/pse_topics.csv", index=False)
print(f"\nSaved topic_summary.csv and pse_topics.csv")

# ── Topics over time — the headline chart ─────────────────────────────────────
# Shows which themes intensified across the conflict timeline.
print("Building topics-over-time...")
timestamps = df["Date"].tolist()
tot = topic_model.topics_over_time(docs, timestamps, nr_bins=24)

fig = topic_model.visualize_topics_over_time(tot, top_n_topics=8)
fig.write_html(f"{BASE}/topics_over_time.html")
print(f"Saved topics_over_time.html")

# ── Cross-check: do topics line up with hand-coded flags? ─────────────────────
# Portfolio gold — shows the unsupervised model rediscovered your regex themes.
print("\nTopic vs. hand-coded flag overlap (mean flag rate per topic):")
flag_cols = [c for c in ["protected_entity", "residential_strike",
                         "children_affected", "repeat_target_text"]
             if c in df.columns]
if flag_cols:
    overlap = df.groupby("topic_reduced")[flag_cols].mean().round(2)
    print(overlap.head(15).to_string())
    overlap.to_csv(f"{BASE}/topic_flag_overlap.csv")
    print("Saved topic_flag_overlap.csv")

print("\nDone.")
