import pandas as pd

# 1. Load your original labels catalog and your injection truth catalog
labels_df = pd.read_csv("train_labels.csv")
truth_df = pd.read_csv("train_truth.csv")

# 2. Merge them together on kepid
review_df = pd.merge(labels_df, truth_df, on="kepid", how="outer")

# 3. Clean up missing values (if a star wasn't in truth_df, injected is 0)
if "injected" in review_df.columns:
    review_df["injected"] = review_df["injected"].fillna(0).astype(int)
else:
    review_df["injected"] = 0

if "label" in review_df.columns:
    review_df["label"] = review_df["label"].fillna(0).astype(int)

# 4. OVERWRITE / UPDATE: If injected == 1, force label to be 1
review_df.loc[review_df["injected"] == 1, "label"] = 1

# 5. Save the updated answer key to train_review.csv
review_df.to_csv("train_review.csv", index=False)

print("=" * 50)
print("🎉 train_review.csv SUCCESSFULLY GENERATED!")
print(f"Total stars in catalog: {len(review_df)}")
print(f"Total positive planet labels (label=1): {(review_df['label'] == 1).sum()}")
print("=" * 50)

# Verify star 5350800
check_star = review_df[review_df["kepid"] == 5350800]
if not check_star.empty:
    print("\nVerified star 5350800 in train_review.csv:")
    print(check_star[["kepid", "label", "injected"]])