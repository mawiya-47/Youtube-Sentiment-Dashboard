"""
01_data_understanding_eda.py
-----------------------------
Phase 2 & 3: Data Understanding and Exploratory Data Analysis.

Run as a script (not a notebook file) so it is reproducible in CI, but every
section below corresponds to what would be one notebook cell + one markdown
explanation cell.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from preprocessing import clean_text

sns.set_theme(style="whitegrid")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "youtube_comments.csv")
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_comments.csv")

# ---------------------------------------------------------------------------
# PHASE 2: DATA UNDERSTANDING
# ---------------------------------------------------------------------------
print("=" * 70)
print("PHASE 2: DATA UNDERSTANDING")
print("=" * 70)

df = pd.read_csv(RAW_PATH)
print(f"\nDataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")

print(f"\nMissing values:\n{df.isna().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")

print(f"\nClass distribution (raw counts):\n{df['Sentiment'].value_counts()}")
print(f"\nClass distribution (%):\n{(df['Sentiment'].value_counts(normalize=True) * 100).round(2)}")

# Note on dataset quality: unlike the "already fully cleaned" description,
# the actual uploaded file contains 44 missing comments and 531 duplicate
# rows out of 18,409. We handle these explicitly rather than assuming a
# pristine dataset — this is standard practice: never trust a data
# description over the data itself.
missing_before = df.isna().sum().sum()
dupes_before = df.duplicated().sum()

df = df.dropna(subset=["Comment", "Sentiment"]).copy()
df = df.drop_duplicates().copy()
df["Sentiment"] = df["Sentiment"].str.strip().str.lower()
df = df[df["Sentiment"].isin(["positive", "negative", "neutral"])].reset_index(drop=True)

print(f"\nAfter cleaning: removed {missing_before} missing values and {dupes_before} duplicate rows")
print(f"Final dataset shape: {df.shape}")

df["comment_length_chars"] = df["Comment"].astype(str).apply(len)
df["comment_length_words"] = df["Comment"].astype(str).apply(lambda x: len(x.split()))

print(f"\nStatistical summary of comment length (characters):\n{df['comment_length_chars'].describe()}")
print(f"\nStatistical summary of comment length (words):\n{df['comment_length_words'].describe()}")
print(f"\nUnique comments: {df['Comment'].nunique()} / {len(df)}")

# Business understanding note (printed for the report, not derived from data)
print("""
Business Understanding:
- Source: YouTube video comments (mixed topics: tech products, tutorials, etc.)
- Use case: automatically triage/monitor audience sentiment at scale, flag
  negative sentiment spikes, and quantify content reception without manual review.
- Class imbalance (positive >> neutral > negative) reflects a real-world prior:
  people comment more when praising content. Models and metrics must account
  for this (macro F1, not just accuracy).
""")

# ---------------------------------------------------------------------------
# PHASE 3: EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------------
print("=" * 70)
print("PHASE 3: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

PALETTE = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#3498db"}
order = ["positive", "neutral", "negative"]

# 1. Class distribution: bar chart
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Sentiment", order=order, palette=PALETTE)
plt.title("Class Distribution (Count Plot)")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_class_distribution_bar.png", dpi=120)
plt.close()

# 2. Class distribution: pie chart
plt.figure(figsize=(6, 6))
counts = df["Sentiment"].value_counts().reindex(order)
plt.pie(counts, labels=order, autopct="%1.1f%%", colors=[PALETTE[c] for c in order], startangle=90)
plt.title("Class Distribution (Pie Chart)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_class_distribution_pie.png", dpi=120)
plt.close()

# 3. Comment length histogram
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="comment_length_words", bins=50, kde=True, color="#3498db")
plt.title("Comment Length Distribution (Words)")
plt.xlabel("Word Count")
plt.xlim(0, df["comment_length_words"].quantile(0.99))
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_comment_length_histogram.png", dpi=120)
plt.close()

# 4. Comment length boxplot by class
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Sentiment", y="comment_length_words", order=order, palette=PALETTE, showfliers=False)
plt.title("Comment Length by Sentiment (Boxplot)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_comment_length_boxplot.png", dpi=120)
plt.close()

# 5. Violin plot
plt.figure(figsize=(7, 5))
sns.violinplot(data=df, x="Sentiment", y="comment_length_words", order=order, palette=PALETTE)
plt.title("Comment Length by Sentiment (Violin Plot)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_comment_length_violin.png", dpi=120)
plt.close()

# 6. Character length distribution
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="comment_length_chars", bins=50, color="#9b59b6")
plt.title("Character Count Distribution")
plt.xlabel("Character Count")
plt.xlim(0, df["comment_length_chars"].quantile(0.99))
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_char_distribution.png", dpi=120)
plt.close()

# 7. Clean text for word-level analysis (cached to processed file)
print("\nCleaning text for word-frequency / word-cloud analysis (this can take a minute)...")
df["clean_comment"] = df["Comment"].apply(clean_text)
df = df[df["clean_comment"].str.len() > 0].reset_index(drop=True)

os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
df.to_csv(PROCESSED_PATH, index=False)
print(f"Saved cleaned dataset to {PROCESSED_PATH} ({len(df)} rows)")

# 8. Most / least common words overall
all_words = " ".join(df["clean_comment"]).split()
word_freq = pd.Series(all_words).value_counts()
print(f"\nTop 20 most common words:\n{word_freq.head(20)}")
print(f"\n20 least common words (appear once):\n{word_freq[word_freq == 1].tail(20)}")

plt.figure(figsize=(9, 6))
word_freq.head(20).plot(kind="barh", color="#16a085")
plt.gca().invert_yaxis()
plt.title("Top 20 Most Frequent Words (All Classes)")
plt.xlabel("Frequency")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07_top_words_overall.png", dpi=120)
plt.close()

# 9. Word clouds per class
for sentiment in order:
    text_blob = " ".join(df[df["Sentiment"] == sentiment]["clean_comment"])
    if not text_blob.strip():
        continue
    wc = WordCloud(width=900, height=500, background_color="white",
                    colormap="viridis", max_words=150).generate(text_blob)
    plt.figure(figsize=(10, 5.5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud — {sentiment.capitalize()} Comments")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/08_wordcloud_{sentiment}.png", dpi=120)
    plt.close()

print(f"\nAll EDA figures saved to: {FIG_DIR}")
print("EDA complete.")
