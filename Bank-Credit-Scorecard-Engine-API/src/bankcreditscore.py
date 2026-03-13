"""
Bank Credit Scorecard Calculation Module
Contains the core logical model for generating credit scores using 
Weight of Evidence (WOE) and Information Value (IV) transformations.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class BankScoreCalculating:
    """
    A logical Credit Scorecard model employing Logistic Regression.
    Utilizes Weight of Evidence (WOE) scaling to classify and score loan applications.
    """
    
    def __init__(self, base_score: int = 600, base_odds: int = 50, pdo: int = 20, epsilon: float = 1e-6):
        self.base_score = base_score
        self.base_odds = base_odds
        self.pdo = pdo
        self.epsilon = epsilon
        self.woe_map = {}
        self.model = LogisticRegression()

        # Compute foundational scorecard scaling parameters
        self.factor = self.pdo / np.log(2)
        self.offset = self.base_score - (self.factor * np.log(self.base_odds))

    def _calculate_woe(self, df: pd.DataFrame, categorical_col: str) -> dict:
        """
        Helper method: Calculates Weight of Evidence (WOE) and Information Value (IV)
        mapping for a specific categorical column based on the target distribution.
        """
        df = df.copy()
        
        # Calculate target totals
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

        # Calculate Information Value (IV) to measure predictive power
        stats['iv'] = (stats['good_dist'] - stats['bad_dist']) * stats['woe']

        print(f"\n--- Information Value (IV) for {categorical_col} ---")
        print(stats['iv'].to_string())

        return stats['woe'].to_dict()

    def _prepare_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Helper method: Discretizes continuous variables into categorical bins and 
        maps them to their corresponding WOE values.
        """
        df_processed = df.copy()

        # Binning continuous variables into predefined categories based on business logic
        df_processed['age_cat'] = pd.cut(df_processed['age'], bins=[0, 20, 50, np.inf], labels=['Young', 'Mid', 'Old'])
        
        df_processed['emp_length_cat'] = pd.cut(df_processed['employment_length'], bins=[-1, 0, 10, 30, np.inf], labels=['None', 'New', 'Mid', 'High'])
        
        df_processed['income_cat'] = np.where(df_processed['income'] <= 100, 'Low', 'High')
        df_processed['dti_cat'] = np.where(df_processed['dti'] <= 0.5, 'Low', 'High')
        
        df_processed['num_credit_lines_cat'] = pd.cut(df_processed['num_credit_lines'], bins=[0, 5, 15, np.inf], labels=['Low', 'Mid', 'High'])

        # Mapping dictionary to link the feature to its corresponding categorical column
        feature_to_category_map = {
            'age': 'age_cat',
            'education': 'education',
            'employment_length': 'emp_length_cat',
            'income': 'income_cat',
            'dti': 'dti_cat',
            'num_credit_lines': 'num_credit_lines_cat',
            'previous_default': 'previous_default',
            'housing': 'housing'
        }

        # Initialize an empty DataFrame to store the final WOE features
        X_woe = pd.DataFrame(index=df_processed.index)

        for feature, cat_col in feature_to_category_map.items():
            if is_train:
                # Calculate and store the WOE mapping using the target variable during training
                woe_dict = self._calculate_woe(df_processed, cat_col)
                self.woe_map[feature] = woe_dict

            # Apply the WOE mapping for both training and inference (predict) phases
            # Missing mappings (unseen categories) are filled with 0 (neutral WOE)
            X_woe[f"{feature}_woe"] = df_processed[cat_col].map(self.woe_map[feature]).astype(float).fillna(0)

        return X_woe 

    def fit(self, df: pd.DataFrame):
        """
        Trains the scorecard by discretizing features, computing WOE, 
        and fitting the underlying logistic regression model.
        """
        df_train = df.copy()
        y = df_train['target']
        
        # Prepare features (defaults to is_train=True)
        X = self._prepare_features(df_train, is_train=True)

        # Fit the logistic regression model
        self.model.fit(X, y)

        print("\n==========================================")
        print("         MODEL TRAINING COMPLETE          ")
        print("==========================================")
        print(f"Model Intercept : {self.model.intercept_[0]:.4f}")
        print(f"Model Coefs     : {self.model.coef_[0]}")
        print(f"WOE Mappings    : {self.woe_map}\n")

        return self

    def predict(self, df: pd.DataFrame) -> int:
        """
        Evaluates new applicant characteristics and returns a final Credit Score.
        """
        # Prepare features for inference (is_train=False ensures we don't recalculate WOE)
        X = self._prepare_features(df, is_train=False)

        # Calculate Log Odds utilizing the Logistic Regression intercept and coefficients
        log_odds = self.model.intercept_[0] + np.dot(X, self.model.coef_[0])

        # Compute final scorecard credit score using scaling factors
        credit_score = self.offset - (self.factor * log_odds)[0]

        return int(round(credit_score))