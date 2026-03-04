# 🏦 Birbank Credit Scorecard Model & API

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Credit Risk](https://img.shields.io/badge/Domain-Credit_Risk-success?style=for-the-badge)

## 📖 Project Overview
The **Birbank Credit Scorecard Model & API** is a robust, production-simulated risk assessment engine designed for the financial sector. Built as part of an advanced data science learning process *(Mentored by Gemini 3.1)*, this project demonstrates how to transform raw applicant data into a standardized credit score using industry-best practices.

Instead of relying on black-box algorithms, this project utilizes **Weight of Evidence (WOE)** binning and **Logistic Regression** to create a highly interpretable scorecard. The model is wrapped in a high-performance **FastAPI** application, allowing downstream systems to instantly request loan approval decisions based on real-time calculations.

## ✨ Key Features
*   **Weight of Evidence (WOE) Engineering:** Implements dynamic WOE calculation to bin continuous variables (Age, Income) into statistically significant risk categories, naturally handling non-linear relationships.
*   **Standard Scorecard Scaling:** Converts raw logistic regression log-odds into a recognizable credit score format utilizing standard parameters: Base Score (600), Base Odds (50:1), and Points to Double Odds (PDO = 20).
*   **Asynchronous REST API:** Deploys the trained scorecard model directly via FastAPI, enabling instantaneous, concurrent loan evaluations.
*   **Data Validation:** Utilizes `Pydantic` schemas to strictly validate incoming JSON payloads, preventing API errors from malformed requests.
*   **Automated Business Logic:** Instantly returns a definitive `Approved` or `Rejected` decision based on a predefined business cut-off score of **550**.

## 📊 Data Description
The model trains on a localized sample dataset representing historical loan applicants. It assesses risk based on two primary features mapped against a binary default target:
*   **Features:**
    *   `age`: Applicant's age in years. Binned internally into `young` (≤ 30) and `old` (> 30).
    *   `income`: Applicant's monthly income. Binned internally into `low` (≤ 600) and `high` (> 600).
*   **Target:** `target` (1 = Bad Credit / Default, 0 = Good Credit / Repaid).

## 🛠️ Project Architecture

```text
├── main.py                                    # Model class, training logic, and FastAPI endpoints
└── README.md                                  # Project documentation
```

## 🚀 Installation & Prerequisites

To run this API locally, you will need Python 3.8+ installed. 

1. **Clone the repository:**
   ```bash
   git clone [Insert Repository Link Here]
   cd [Insert Repository Directory Name]
   ```

2. **Install the required dependencies:**
   It is highly recommended to use a virtual environment.
   ```bash
   pip install fastapi uvicorn scikit-learn pandas numpy pydantic
   ```

## 💻 Usage / How to Run

### Step 1: Launch the FastAPI Server
The model automatically trains itself on startup and exposes the endpoint. Start the server using Uvicorn:

```bash
uvicorn main:app --reload
```
*Expected Terminal Output during startup:*
> `--- Training Complete ---`
> `Model Intercept : [Value]`
> `Model Coefs     : [Values]`
> `WOE Mappings    : {...}`

### Step 2: Test the API Endpoint
Once the server is running (typically on `http://127.0.0.1:8000`), you can interact with the API in two ways:

**Option A: Interactive Swagger UI (Recommended)**
Open your browser and navigate to: `http://127.0.0.1:8000/docs`. You can use the built-in interface to send test requests.

**Option B: Terminal (cURL)**
Send a POST request simulating a customer application:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/apply_loan' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 28,
  "income": 1200
}'
```

## 📈 Results / Outputs

When a successful request is made, the API processes the features through the WOE map, applies the Logistic Regression coefficients, scales it to a credit score, and returns a clear JSON response. 

**Example API Response:**
```json
{
  "Status": "Success",
  "Customer_Profile": {
    "Age": 28,
    "Income": 1200
  },
  "Score": 612,
  "Decision": "Approved"
}
```
*(Note: Since the calculated score of 612 is greater than the strict cut-off of 550, the loan is Approved).*

## 🤝 Contributing
This project is an excellent sandbox for learning credit risk modeling! If you'd like to contribute:
1. Fork the Repository
2. Create your Feature Branch (`git checkout -b feature/AddGiniCoefficient`)
3. Commit your Changes (`git commit -m 'Add Model Gini/AUC metrics on startup'`)
4. Push to the Branch (`git push origin feature/AddGiniCoefficient`)
5. Open a Pull Request

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

---

## 📬 Contact
**Selim Najaf**

*   **LinkedIn:** [linkedin.com/in/selimnajaf-data-analyst](https://www.linkedin.com/in/selimnajaf/)
*   **GitHub:** [github.com/SelimNajaf](https://github.com/SelimNajaf)

*Developed as a continuous learning initiative in advanced Data Science and ML Engineering.*
