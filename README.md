# 🏦 Bank Credit Scorecard Engine & API

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FinTech](https://img.shields.io/badge/Domain-FinTech_%7C_Credit_Risk-success?style=for-the-badge)

## 📖 Project Overview
In the highly regulated financial sector, machine learning models cannot be "black boxes"—they must be fully interpretable. The **Bank Credit Scorecard Engine** is a production-grade ML solution that evaluates customer loan applications using industry-standard **Weight of Evidence (WOE)** and **Information Value (IV)** transformations.

Rather than relying on standard algorithms, this project features a fully custom Object-Oriented model (`BankScoreCalculating`) built on top of Scikit-Learn's Logistic Regression. It translates complex log-odds into a traditional, human-readable Credit Score (scaled to a baseline of 600). The finalized scorecard is then deployed as a high-performance **FastAPI** web service to instantly process incoming loan requests and render mathematical approval decisions.

## ✨ Key Features
*   **Advanced Grouped Imputation:** Cleans incomplete data not by using generic global averages, but by calculating contextual means (e.g., imputing missing `employment_length` based on the applicant's `age` bracket, and `credit_score` based on `income` deciles).
*   **WOE & IV Engineering:** Automatically bins continuous variables and computes Information Value (IV) to measure the predictive power of each feature. Replaces raw values with Weight of Evidence (WOE) to elegantly handle non-linear relationships and missing data during inference.
*   **Custom Scikit-Learn Estimator:** Encapsulates the entire scorecard logic—binning, WOE mapping, fitting, and mathematical scaling—into a reusable Python class.
*   **Standardized Score Scaling:** Translates Logistic Regression outputs into recognizable credit scores using standardized risk parameters: Base Score (600), Base Odds (50:1), and Points to Double Odds (PDO = 20).
*   **FastAPI Deployment:** Wraps the serialized `.joblib` model in a highly responsive REST API, using `Pydantic` to strictly validate demographic and financial payloads before applying a hard-coded business approval cutoff of **550**.

## 📊 Data Description
The model is trained on a localized credit risk dataset (`credit_risk_dataset.csv`) containing 1,000 applicant records.

**Input Features:**
*   **Demographics:** `age`, `education`, `housing` (Rent, Own, Mortgage)
*   **Financials:** `income`, `dti` (Debt-to-Income ratio), `credit_score`
*   **Credit History:** `employment_length`, `num_credit_lines`, `previous_default`

**Target Variable:** 
*   `target`: Binary indicator (1 = Default / Bad Credit, 0 = Repaid / Good Credit)

## 🛠️ Project Architecture

```text
├── src/
│   └── bankcreditscore.py             # Core OOP logic: WOE/IV computation and Scorecard Model
├── train/
│   └── train_model.py                 # EDA, Grouped Imputation, Model Training & Export
├── api/
│   └── main.py                        # FastAPI application and prediction endpoint
├── dataset/
│   └── credit_risk_dataset.csv        # Raw dataset [Not included, download required]
├── model/
│   └── scorecard_model.joblib         # Serialized Custom Scorecard (Generated Output)
└── README.md                          # Project documentation
```

## 🚀 Installation & Prerequisites

To run this pipeline and API locally, ensure you have Python 3.8+ installed.

1. **Clone the repository:**
   ```bash
   git clone [Insert Repository Link Here]
   cd [Insert Repository Directory Name]
   ```

2. **Install the required dependencies:**
   It is highly recommended to use a virtual Python environment.
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn fastapi uvicorn joblib pydantic
   ```

3. **Add the Dataset:**
   Ensure `credit_risk_dataset.csv` is placed inside the `dataset/` directory.

## 💻 Usage / How to Run

### Step 1: Train the Scorecard Model
Execute the training script. This will generate EDA plots, perform grouped data imputation, calculate Information Values (IV), train the Logistic Regression weights, and export the custom pipeline.
*(Note: Close the EDA plot windows when they appear to allow the script to continue).*

```bash
python train/train_model.py
```

### Step 2: Launch the FastAPI Server
Once `scorecard_model.joblib` is generated in the `model/` folder, spin up the API using Uvicorn.

```bash
uvicorn api.main:app --reload
```

### Step 3: Test the Loan Application Endpoint
Navigate to `http://127.0.0.1:8000/docs` to use the interactive Swagger UI, or simulate a live loan application via cURL:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/apply_loan' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 28,
  "education": "Bachelor",
  "employment_length": 3.5,
  "income": 85000.0,
  "dti": 0.45,
  "num_credit_lines": 4,
  "previous_default": 0,
  "housing": "Rent"
}'
```

## 📈 Results & Business Interpretability

Unlike standard algorithms, a WOE scorecard is fully transparent. During training, the script outputs the exact predictive power (Information Value) of every feature. For example, the terminal output reveals that having a `Mortgage` is highly predictive of good credit (high IV), whereas `previous_default` severely damages the score.

**Model Transparency Output:**
> `Model Intercept : -4.2568`  
> `WOE Mappings: {'housing': {'Mortgage': 12.64, 'Own': 0.41, 'Rent': -0.72}, ...}`

### API Decision Logic
The deployed FastAPI evaluates the incoming customer profile, maps it against the trained WOE bins, calculates the Log-Odds, and scales it. If the final score is **≥ 550**, the loan is Approved.

**Example API Response:**
```json
{
  "status": "success",
  "score": 583.0,
  "decision": "Approved"
}
```

## 🤝 Contributing
Contributions are highly encouraged! To further optimize this project:
1. Fork the repository
2. Create your Feature Branch (`git checkout -b feature/AddPopulationStabilityIndex`)
3. Commit your Changes (`git commit -m 'Add PSI calculator to monitor data drift'`)
4. Push to the Branch (`git push origin feature/AddPopulationStabilityIndex`)
5. Open a Pull Request

## 📜 License
This project is open-source and available under the MIT License. See `LICENSE` for more information.

## 📬 Contact
**Selim Najaf**

*   **LinkedIn:** [linkedin.com/in/selimnajaf-data-analyst](https://www.linkedin.com/in/selimnajaf/)
*   **GitHub:** [github.com/SelimNajaf](https://github.com/SelimNajaf)

*Developed as a continuous learning initiative in advanced Data Science and ML Engineering.*
