import pandas as pd
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import os

# Load and Clean Data
df = pd.read_csv("archive/loan/loan.csv", low_memory=False)

# Drop columns with >80% missing
df.dropna(thresh=len(df) * 0.2, axis=1, inplace=True)

# Filter for relevant loan statuses
statuses = [
    "Fully Paid", "Charged Off", "Current",
    "Late (31-120 days)", "Late (16-30 days)", "In Grace Period"
]
df = df[df["loan_status"].isin(statuses)]

# Convert date columns (day-month-year format)
df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
df["last_pymnt_d"] = pd.to_datetime(df["last_pymnt_d"], errors="coerce")
df.dropna(subset=["issue_d", "last_pymnt_d"], inplace=True)

# Calculate loan duration in months
df["duration"] = (df["last_pymnt_d"] - df["issue_d"]).dt.days // 30

# Encode event: 1 = Charged Off (defaulted), 0 = otherwise (censored)
df["event_observed"] = df["loan_status"].apply(lambda x: 1 if x == "Charged Off" else 0)

# Convert grade A–G to numeric 
grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
df["grade"] = df["grade"].map(grade_map)

# Clean emp_length column (convert "10+ years" → 10, "<1 year" → 0, etc.)
df["emp_length"] = (
    df["emp_length"]
    .str.extract(r"(\d+)")
    .fillna(0)
    .astype(int)
)

# One-hot encode home_ownership and verification_status
df = pd.get_dummies(df, columns=["home_ownership", "verification_status"], drop_first=True)

# Dynamically detect all dummy columns
home_dummies = [col for col in df.columns if col.startswith("home_ownership_")]
verif_dummies = [col for col in df.columns if col.startswith("verification_status_")]

# Select Features and Fit Model

features = [
    "duration", "event_observed",
    "funded_amnt", "int_rate", "installment",
    "grade", "emp_length", "annual_inc"
] + verif_dummies

df_cox = df[features].dropna()

# Fit the Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(df_cox, duration_col="duration", event_col="event_observed")

cph.print_summary()
