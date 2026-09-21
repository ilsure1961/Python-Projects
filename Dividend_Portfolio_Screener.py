import os
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Load
# --------------------------------------------------

file_path = r"C:\Desktop\Python\Source_DVD.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip().str.lstrip("#")
print("Columns detected:")
print(df.columns.tolist())

# --------------------------------------------------
# 2. Clean numeric fields
# --------------------------------------------------

percent_columns = [
    # "Number_Y_Increase",  # remove this; it's a count, not a percentage
    "DVD_5Y_Gr",
    "DVD_Yield",
    "EPS_Gr",
    "Rev_Gr",
    "Payout",
    "ROIC",
    "DE",
]

# Rename source column to expected name
if "Yrs_DVD_Increase" in df.columns and "Number_Y_Increase" not in df.columns:
    df.rename(columns={"Yrs_DVD_Increase": "Number_Y_Increase"}, inplace=True)

# Parse it separately - no /100 conversion
df["Number_Y_Increase"] = pd.to_numeric(
    df["Number_Y_Increase"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip(),
    errors="coerce",
)

# Ensure numeric
df["DVD_Cash_Cover"] = pd.to_numeric(df["DVD_Cash_Cover"], errors="coerce")

# Compute Payout ratio from DPS and EPS if not present
if "Payout" not in df.columns:
    df["EPS_num"] = pd.to_numeric(
        df["EPS"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )
    df["DPS_num"] = pd.to_numeric(
        df["DPS"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )
    df["Payout"] = df["DPS_num"] / df["EPS_num"]

# Add placeholder columns for missing metrics
for missing_col in ["ROIC", "DE"]:
    if missing_col not in df.columns:
        df[missing_col] = np.nan

# Convert to decimals if needed
for col in percent_columns:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert to decimals if needed
        if df[col].dropna().max() > 2:
            df[col] = df[col] / 100

# --------------------------------------------------
# 3. Validate required columns
# --------------------------------------------------

required_cols = [
    "Number_Y_Increase",
    "DVD_5Y_Gr",
    "DVD_Yield",
    "EPS_Gr",
    "Rev_Gr",
    "Payout",
    "DVD_Cash_Cover",
    "ROIC",
    "DE",
    "Sector",
]

missing = [col for col in required_cols if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# --------------------------------------------------
# 4. Scoring model
# --------------------------------------------------


def score_row(row):
    score = 0

    # Dividend / Growth Profile
    score += 0.15 if row["Number_Y_Increase"] >= 20 else 0
    score += 0.15 if row["DVD_5Y_Gr"] >= 0.10 else 0
    score += 0.05 if row["DVD_Yield"] >= 0.02 else 0

    # Fundamentals
    score += 0.20 if row["EPS_Gr"] >= 0.05 else 0
    score += 0.10 if row["Rev_Gr"] >= 0.03 else 0

    # Dividend Safety / Sustainability
    score += (
        0.10
        if pd.notna(row["Payout"]) and row["Payout"] < 0.60
        else 0
    )
    score += (
        0.10
        if pd.notna(row["DVD_Cash_Cover"]) and row["DVD_Cash_Cover"] >= 2
        else 0
    )
    score += (
        0.05
        if pd.notna(row["ROIC"]) and row["ROIC"] >= 0.10
        else 0
    )
    score += (
        0.10
        if pd.notna(row["DE"]) and row["DE"] <= 1.00
        else 0
    )

    return score


df["Score"] = df.apply(score_row, axis=1)

# --------------------------------------------------
# 5. Global rank (1 best -> 5 worst)
# --------------------------------------------------

try:
    df["Rank"] = pd.qcut(
        df["Score"],
        q=5,
        labels=[5, 4, 3, 2, 1],
        duplicates="drop",
    )
except ValueError:
    df["Rank"] = pd.cut(
        df["Score"],
        bins=[-0.01, 0.20, 0.40, 0.60, 0.80, 1.00],
        labels=[5, 4, 3, 2, 1],
    )

df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce").fillna(5).astype(int)

# --------------------------------------------------
# 6. Sector ranking
# --------------------------------------------------

df["Sector_Rank"] = (
    df.groupby("Sector")["Score"].rank(method="first", ascending=False)
)

# --------------------------------------------------
# 7. Build diversified top 10
# --------------------------------------------------

SECTOR_LIMIT = 2  # max stocks per sector (adjust if needed)

top10 = (
    df.sort_values(by="Score", ascending=False)
    .groupby("Sector")
    .head(SECTOR_LIMIT)  # limit per sector
    .sort_values(by="Score", ascending=False)
    .head(10)  # final top 10
)

# --------------------------------------------------
# 8. Output
# --------------------------------------------------

print("\nTop 10 Stocks (Sector Diversified):")
print(top10[["Sector", "Score", "Rank", "Sector_Rank"]])

print("\nSector Distribution:")
print(top10["Sector"].value_counts())

# --------------------------------------------------
# 9. Save files
# --------------------------------------------------

df_sorted = df.sort_values(by=["Score"], ascending=False)

df_sorted.to_csv("Ranked_Stocks.csv", index=False)
top10.to_csv("Top10_Stocks_BySector.csv", index=False)

print("\nFiles created:")
print("1. Ranked_Stocks.csv")
print("2. Top10_Stocks_BySector.csv")