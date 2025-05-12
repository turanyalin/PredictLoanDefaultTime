Predicting Loan Default Time with Survival Analysis

This project uses a Cox Proportional Hazards Model to predict how long a loan will survive before defaulting based on borrower and loan characteristics. The dataset is from LendingClub and includes hundreds of thousands of loans.

Features
Cleaned and preprocessed LendingClub dataset
Applied Kaplan-Meier estimator to analyze general survival curves
Built and interpreted a Cox Proportional Hazards Model
Identified the impact of key variables on default risk:
Interest Rate
Credit Grade
Loan Amount
Employment Length
Income Verification
Produced interpretable outputs for financial risk assessment

Requirements
Install dependencies:

pip install -r requirements.txt
Main libraries:

pandas
lifelines
matplotlib
Data Source
This project uses LendingClub loan data from Kaggle:
Download the dataset here
https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset
After downloading:

Place the CSV at: archive/loan/loan.csv
The CSV is not included in this repo due to GitHub’s 100MB file limit.

Key Insights
Loans with higher interest rates and lower credit grades default faster.
Employment length and higher income slightly reduce risk.
Survival modeling provides a nuanced view: not just if a loan defaults, but when.

How to Run
python3 CreditRisk.py

License
This project is for educational and research purposes only.
