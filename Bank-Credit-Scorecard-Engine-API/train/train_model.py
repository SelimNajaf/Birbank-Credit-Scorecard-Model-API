"""
Bank Credit Scorecard - Training Pipeline
This script loads credit risk data, performs Exploratory Data Analysis (EDA),
handles missing values through grouped imputation, trains an internal logistic 
regression scorecard, and exports the model for deployment.
"""

import os
import sys
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 0. SYSTEM PATH CONFIGURATION
# ==========================================
# Dynamically add the project root to sys.path so the 'src' package 
# can be discovered regardless of where the script is executed.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import your custom scorecard model class
from src.bankcreditscore import BankScoreCalculating

# ==========================================
# 1. DATA LOADING & EXPLORATION
# ==========================================
FILE_PATH = 'dataset/credit_risk_dataset.csv'

try:
    df = pd.read_csv(FILE_PATH)
    print(f"Successfully loaded dataset from '{FILE_PATH}'")
except FileNotFoundError:
    print(f"Error: Dataset '{FILE_PATH}' not found. Please ensure it is in the correct directory.")
    sys.exit(1)

print("\n--- DataFrame Head ---")
print(df.head())

print("\n--- Missing Values Before Imputation ---")
print(df.isnull().sum())

print("\n--- Data Statistics ---")
print(df.describe())

print("\n--- DataFrame Information ---")
df.info()

# ==========================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.to_list()

print("\nGenerating Correlation Heatmap. (Note: Close the plot window to continue...)")
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Numeric Features Correlation")
plt.show()

print("Generating Feature Histograms. (Note: Close the plot window to continue...)")
df.hist(figsize=(12, 8), bins=15)
plt.suptitle("Feature Distributions")
plt.show()

# ==========================================
# 3. DATA PREPROCESSING & IMPUTATION
# ==========================================
print("\n--- Handling Missing Values ---")

# 1. Impute employment_length grouped by age, fallback to global mean
df['employment_length'] = df['employment_length'].fillna(
    df.groupby('age')['employment_length'].transform('mean')
)
df['employment_length'] = df['employment_length'].fillna(df['employment_length'].mean())

# 2. Impute income grouped by age
df['income'] = df['income'].fillna(df.groupby('age')['income'].transform('mean'))

# 3. Impute credit_score using income bins, then fallback to age groups
df['income_bin'] = pd.qcut(df['income'], 10, duplicates='drop')
df['credit_score'] = df['credit_score'].fillna(df.groupby('income_bin')['credit_score'].transform('mean'))
df.drop('income_bin', axis=1, inplace=True)
df['credit_score'] = df['credit_score'].fillna(df.groupby('age')['credit_score'].transform('mean'))

# 4. Impute debt-to-income (dti) ratio grouped by age
df['dti'] = df['dti'].fillna(df.groupby('age')['dti'].transform('mean'))

print("\n--- Missing Values After Imputation ---")
print(df.isnull().sum())

# ==========================================
# 4. MODEL TRAINING
# ==========================================
print("\n--- Training Scorecard Model ---")
# Initialize and train the custom scorecard model
model_instance = BankScoreCalculating()
model_instance.fit(df)

# ==========================================
# 5. EXPORTING THE MODEL
# ==========================================
# Ensure the save directory exists
SAVE_DIR = 'model'
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(SAVE_DIR, 'scorecard_model.joblib')
joblib.dump(model_instance, MODEL_PATH)

print(f"\nTraining complete! Scorecard model successfully saved to '{MODEL_PATH}'")