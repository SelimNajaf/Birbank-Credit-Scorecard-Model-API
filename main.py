"""
Birbank Credit Scorecard Model & API
Trains an internal logistic regression scorecard utilizing Weight of Evidence (WOE) 
and exposes it via an asynchronous FastAPI endpoint for loan approval processing.
"""

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression

# ==========================================
# 1. SAMPLE TRAINING DATA
# ==========================================
training_data = pd.DataFrame({
    'age':[22, 25, 30, 45, 50, 21, 28, 35, 40, 55, 23, 26, 31, 46, 51],
    'income':[300, 400, 800, 1500, 2000, 350, 500, 900, 1600, 2200, 320, 450, 850, 1550, 2100],
    'target':[1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0]  # 1 = Bad Credit, 0 = Good Credit
})


# ==========================================
# 2. SCORECARD MODEL
# ==========================================
class BirbankScorecard:
    """
    A logical Credit Scorecard model employing Logistic Regression.
    Utilizes Weight of Evidence (WOE) scaling to classify and score loan applications.
    """
    
    def __init__(self, base_score=600, base_odds=50, pdo=20, epsilon=1e-6):
        self.base_score = base_score
        self.base_odds = base_odds
        self.pdo = pdo
        self.epsilon = epsilon
        self.woe_map = {}
        self.model = LogisticRegression()

        # Compute foundational scorecard parameters
        self.factor = pdo / np.log(2)
        self.offset = base_score - (self.factor * np.log(base_odds))

    def _calculate_woe(self, df: pd.DataFrame, categorical_col: str) -> dict:
        """Helper method: Calculates WOE mapping for a specific categorical column."""
        df = df.copy()
        
        # Calculate totals
        total_good = (df['target'] == 0).sum()
        total_bad = (df['target'] == 1).sum()

        # Create crosstab distribution
        stats = pd.crosstab(df[categorical_col], df['target'])
        stats.columns = ['good', 'bad']

        # Determine distribution proportions
        stats['good_dist'] = stats['good'] / total_good
        stats['bad_dist'] = stats['bad'] / total_bad
        
        # Calculate actual Weight of Evidence (WOE) applying epsilon to prevent division by zero
        stats['woe'] = np.log((stats['good_dist'] + self.epsilon) / 
                              (stats['bad_dist'] + self.epsilon))

        stats['iv'] = (stats['good_dist'] - stats['bad_dist']) * stats['woe']

        print(f"IV Values: {stats['iv']}")

        return stats['woe'].to_dict()

    def fit(self, df: pd.DataFrame):
        """Trains the scorecard by computing WOE and fitting the logistic model."""
        df_train = df.copy()

        # Binarize/categorize features based on business logic
        df_train['age_cat'] = np.where(df_train['age'] <= 30, 'young', 'old')
        df_train['income_cat'] = np.where(df_train['income'] <= 600, 'low', 'high')

        # Calculate and store WOE mappings internally
        self.woe_map['age'] = self._calculate_woe(df_train, 'age_cat')
        self.woe_map['income'] = self._calculate_woe(df_train, 'income_cat')

        # Apply mapped WOE features directly to training set
        df_train['age_woe'] = df_train['age_cat'].map(self.woe_map['age'])
        df_train['income_woe'] = df_train['income_cat'].map(self.woe_map['income'])

        # Prepare regression variables
        X = df_train[['age_woe', 'income_woe']]
        y = df_train['target']

        # Fit the logistic regression model
        self.model.fit(X, y)

        print("\n--- Training Complete ---")
        print(f"Model Intercept : {self.model.intercept_[0]:.4f}")
        print(f"Model Coefs     : {self.model.coef_[0]}")
        print(f"WOE Mappings    : {self.woe_map}\n")

        return self

    def predict(self, age: int, income: int) -> int:
        """Returns a final Credit Score based on applicant characteristics."""
        
        # Determine category based on input
        age_cat = 'young' if age <= 30 else 'old'
        income_cat = 'low' if income <= 600 else 'high'

        # Map to internally stored WOE values
        age_woe = self.woe_map['age'][age_cat]
        income_woe = self.woe_map['income'][income_cat]

        # Structure input for model prediction
        X = np.array([[age_woe, income_woe]])

        # Calculate Log Odds utilizing LR intercept & coefficients
        log_odds = self.model.intercept_[0] + np.dot(X, self.model.coef_[0])[0]

        # Compute final scorecard credit score
        credit_score = self.offset - (self.factor * log_odds)

        return int(round(credit_score))


# ==========================================
# 3. FASTAPI ENDPOINT SETUP
# ==========================================
# Initialize and train the model immediately upon startup
model_instance = BirbankScorecard()
model_instance.fit(training_data)

# Hard-coded Business Cutoff Value
CUT_OFF = 550

app = FastAPI(title="Birbank Scorecard Engine", version="1.0.0")

class CustomerRequest(BaseModel):
    """Pydantic schema representing incoming customer loan applications."""
    age: int
    income: int

@app.post('/apply_loan')
def apply_loan(customer: CustomerRequest):
    """
    Evaluates a customer's loan application based on Age and Income.
    Returns calculated credit score and ultimate approval decision.
    """
    # Predict credit score dynamically
    score = model_instance.predict(customer.age, customer.income)

    # Determine business decision
    decision = 'Approved' if score >= CUT_OFF else 'Rejected'

    return {
        'Status': 'Success',
        'Customer_Profile': {
            'Age': customer.age,
            'Income': customer.income
        },
        'Score': score,
        'Decision': decision
    }