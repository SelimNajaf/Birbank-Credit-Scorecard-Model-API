"""
Bank Credit Scorecard API
A FastAPI web service that loads a pre-trained internal scorecard model 
utilizing Weight of Evidence (WOE) to evaluate customer loan applications.
"""

import os
import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ==========================================
# 0. SYSTEM PATH CONFIGURATION
# ==========================================
# Required so joblib can successfully unpickle the custom class
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ==========================================
# 1. MODEL LOADING
# ==========================================
MODEL_PATH = 'model/scorecard_model.joblib'

try:
    # Load the pre-trained scorecard model into memory upon startup
    model_instance = joblib.load(MODEL_PATH)
    print(f"Successfully loaded model from '{MODEL_PATH}'")
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    print("Please run 'train_model.py' to generate the model before starting the API.")
    sys.exit(1)

# Hard-coded Business Cutoff Value
CUT_OFF = 550

# ==========================================
# 2. FASTAPI INITIALIZATION & SCHEMAS
# ==========================================
app = FastAPI(
    title="Bank Scorecard Engine", 
    description="Evaluates customer financial profiles to approve or reject loan applications based on calculated credit scores.",
    version="1.0.0"
)

class CustomerRequest(BaseModel):
    """Schema defining the expected JSON payload for incoming loan applications."""
    age: int = Field(..., description="Applicant's age in years")
    education: str = Field(..., description="Highest level of education completed")
    employment_length: float = Field(..., description="Years of employment at current job")
    income: float = Field(..., description="Annual income in local currency")
    dti: float = Field(..., description="Debt-to-Income ratio")
    num_credit_lines: int = Field(..., description="Total number of open credit lines")
    previous_default: int = Field(..., description="History of previous defaults (1 = Yes, 0 = No)")
    housing: str = Field(..., description="Housing status (e.g., 'RENT', 'OWN', 'MORTGAGE')")

class LoanResponse(BaseModel):
    """Schema representing the structure of the API response."""
    status: str
    score: float
    decision: str


# ==========================================
# 3. PREDICTION ENDPOINT
# ==========================================
@app.post('/apply_loan', response_model=LoanResponse)
async def apply_loan(customer: CustomerRequest):
    """
    Receives customer demographic and financial data, calculates their 
    credit score, and returns a loan approval decision.
    """
    try:
        # Step 1: Convert validated Pydantic object into a pandas DataFrame
        df = pd.DataFrame([customer.model_dump()])

        # Step 2: Generate the credit score using the custom scorecard logic
        score = float(model_instance.predict(df))

        # Step 3: Apply the business rule for approval
        decision = "Approved" if score >= CUT_OFF else "Rejected"

        # Step 4: Return formatted response
        return LoanResponse(
            status="success",
            score=score,
            decision=decision
        )
        
    except Exception as e:
        # Gracefully handle and report any unexpected execution errors
        raise HTTPException(status_code=500, detail=f"Scorecard processing error: {str(e)}")