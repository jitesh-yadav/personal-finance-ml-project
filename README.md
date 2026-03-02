# Personal Finance Intelligence Platform

## Table of Contents

- [Overview](#overview)
- [Business Problem](#business-problem)
- [Solution](#solution)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Key Features](#key-features)
- [Machine Learning Models](#machine-learning-models)
- [Results & Impact](#results--impact)
- [Visualizations](#visualizations)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [Contact](#contact)

---

##  Overview

A comprehensive **data science and machine learning platform** that transforms raw financial transaction data into actionable insights. Built with Python, SQL, and advanced ML algorithms, this project demonstrates end-to-end skills in data engineering, statistical modeling, and business intelligence.

**Project Highlights:**
- 📊 Processed and analyzed **1,500+ financial transactions** spanning 5 years
- 🤖 Implemented **5 machine learning models** (supervised, unsupervised, time series)
- 📈 Created **50+ interactive visualizations** using Plotly and Seaborn
- 🗄️ Designed **SQL database** with 55+ optimized queries
- 💡 Generated **actionable insights** for financial decision-making

---

## 💼 Business Problem

**Challenge:** 
Individuals and small businesses struggle to understand their spending patterns, predict future expenses, and identify anomalous transactions without sophisticated analytics tools.

**Impact:**
- Lack of visibility into spending trends
- Inability to forecast cash flow accurately
- Missing opportunities to optimize budgets
- Delayed detection of unusual transactions

**Goal:**
Build an intelligent system that automatically analyzes financial data, predicts future spending, detects anomalies, and provides actionable recommendations.

---

##  Solution

Developed a **3-phase data science pipeline** that delivers:

### Phase 1: Data Foundation
- Cleaned and standardized 1,500 transactions with 100% data quality
- Engineered **27 predictive features** (temporal, statistical, categorical)
- Built normalized SQL database with optimized indexes
- Created reusable data processing scripts

### Phase 2: Machine Learning
- **Random Forest Regressor**: Predicted transaction amounts 
- **XGBoost**: Alternative prediction model with feature importance analysis
- **Isolation Forest**: Detected 62 anomalous transactions 
- **K-Means Clustering**: Identified 2 distinct spending personas
- **ARIMA Time Series**: Forecasted 6-month expenses with confidence intervals

### Phase 3: Business Intelligence
- Built interactive dashboards with real-time filtering
- Automated insight generation and recommendations
- Created executive summary reports
- Designed portfolio-ready visualizations

---

## Technologies Used

### **Languages & Libraries**
```
Python 3.9+          │ Core programming language
├── pandas           │ Data manipulation and analysis
├── NumPy            │ Numerical computing
├── scikit-learn     │ Machine learning algorithms
├── XGBoost          │ Gradient boosting framework
├── statsmodels      │ Statistical modeling (ARIMA)
├── Plotly           │ Interactive visualizations
├── Matplotlib       │ Static plotting
└── Seaborn          │ Statistical data visualization

SQL (SQLite)         │ Database design and queries
Google Colab         │ Development environment
Git/GitHub           │ Version control
```

### **ML Techniques**
- Supervised Learning (Regression)
- Unsupervised Learning (Clustering, Anomaly Detection)
- Time Series Forecasting
- Feature Engineering
- Model Evaluation & Selection

---

## Dataset

**Source:** Personal finance transaction data  
**Size:** 1,500 transactions  
**Time Period:** 2020-01-02 to 2024-12-29 (5 years)  
**Categories:** 10 expense/income categories

### Original Features (5)
- Date
- Transaction Description
- Category
- Amount
- Type (Income/Expense)

### Engineered Features (27 total)
**Temporal:** Year, Month, Quarter, Day, Week, Weekend Flag  
**Statistical:** Rolling averages (7-day, 30-day), Lag features  
**Categorical:** Category encoding, Type encoding  
**Aggregated:** Category averages, Monthly totals, Transaction counts

### Data Quality
-  Zero missing values
-  No duplicate transactions
-  Date range validated
-  Amount ranges verified
-  Category consistency ensured

---

## Project Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     RAW DATA (CSV)                          │
│                  1,500 Transactions                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               PHASE 1: DATA PREPARATION                     │
│  • Data Cleaning & Validation                               │
│  • Feature Engineering (27 features)                        │
│  • SQL Database Design                                      │
│  • Excel Analysis Template                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            PHASE 2: MACHINE LEARNING MODELS                 │
│  ┌──────────────┬──────────────┬──────────────┐            │
│  │   Regression │   Anomaly    │  Clustering  │            │
│  │   (XGBoost,  │  Detection   │  (K-Means)   │            │
│  │  Random F.)  │  (Iso Forest)│              │            │
│  └──────────────┴──────────────┴──────────────┘            │
│  │                   Time Series (ARIMA)      │            │
│  └────────────────────────────────────────────┘            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         PHASE 3: VISUALIZATIONS & INSIGHTS                  │
│  • 50+ Interactive Charts (Plotly)                          │
│  • Executive Dashboards                                     │
│  • Automated Insights Engine                                │
│  • Business Recommendations                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. **Automated Data Pipeline**
- Robust ETL process handling raw financial data
- Feature engineering automation
- Data quality validation checks
- Reusable, modular Python scripts

### 2. **SQL Database System**
- Normalized schema (4 tables, 7 indexes, 4 views)
- 55+ pre-written analytical queries
- Optimized for performance
- Pre-aggregated summary tables

### 3. **Predictive Analytics**
- Transaction amount prediction
- 6-month expense forecasting
- Confidence interval estimation
- Model performance tracking

### 4. **Anomaly Detection**
- Automated flagging of unusual transactions
- Category-wise anomaly analysis
- Real-time alerting capability
- False positive minimization

### 5. **Spending Intelligence**
- Persona-based segmentation
- Category optimization recommendations
- Behavioral pattern recognition
- Seasonal trend analysis

### 6. **Interactive Dashboards**
- Real-time filtering and drill-down
- Mobile-responsive design
- Export-ready visualizations
- Executive summary views

---

## Machine Learning Models

### **Model 1: Random Forest Regressor** BEST MODEL
**Purpose:** Predict transaction amounts  
**Performance:**
- Test R² Score: **0.4771**
- RMSE: **$719.51**
- MAE: **$598.03**

**Top Features:**
1. Category_Avg_Amount (28.4%)
2. Type_Encoded (18.3%)
3. Rolling_7day_Avg (14.9%)

**Business Impact:** Enables accurate budgeting and expense forecasting

---

### **Model 2: XGBoost Regressor**
**Purpose:** Alternative prediction model with gradient boosting  
**Performance:**
- Test R² Score: 0.4050
- RMSE: $767.57
- MAE: $632.48

**Key Insight:** Type_Encoded (Income vs Expense) is most important feature (71%)

---

### **Model 3: Isolation Forest**
**Purpose:** Detect anomalous transactions  
**Results:**
- **62 anomalies detected** (5.07% of transactions)
- Highest anomaly rate: Utilities (10.19%)
- Mean anomaly amount: $1,018 vs $1,004 normal

**Business Impact:** Early fraud detection and error identification

---

### **Model 4: K-Means Clustering**
**Purpose:** Customer segmentation / spending persona analysis  
**Results:**
- **2 distinct personas identified**
  - **Big Spender:** $24,455/month avg (31 months)
  - **Frugal Saver:** $16,175/month avg (29 months)
- Silhouette Score: 0.1549

**Business Impact:** Personalized financial advice and targeted recommendations

---

### **Model 5: ARIMA Time Series**
**Purpose:** Forecast future monthly expenses  
**Configuration:** ARIMA(0,1,1)  
**Performance:**
- RMSE: $5,262.59
- MAPE: 24.67%
- 6-month forecast: **$17,719.54/month average**

**Business Impact:** Cash flow planning and budget preparation

---

## Results & Impact

### **Key Findings**

#### **Critical Insight: Negative Savings Rate**
- **Savings Rate: -67.17%** (expenses exceed income)
- Total Income: $734,087
- Total Expenses: $1,227,194
- **Net Loss: -$493,107**

**Recommendation:** Reduce spending by 15% in top 3 categories to achieve positive cash flow

---

#### **Top Spending Categories**
| Rank | Category | Total Spent | % of Budget |
|------|----------|-------------|-------------|
| 1 | Travel | $169,497.79 | 13.81% |
| 2 | Rent | $162,075.39 | 13.21% |
| 3 | Food & Drink | $159,493.39 | 13.00% |

**Insight:** Top 3 categories account for 40% of all spending

---

#### **Anomaly Detection Results**
- 62 unusual transactions identified
- Most anomalies in: Utilities (16), Food & Drink (11)
- Average anomaly amount: $1,018 (vs $1,004 normal)

**Action Item:** Review flagged Utilities transactions for billing errors

---

#### **Spending Personas**
**Big Spender Persona:**
- Average: $24,455/month
- Top category: Salary ($3,477)
- Transaction frequency: 23/month

**Frugal Saver Persona:**
- Average: $16,175/month
- Top category: Travel ($2,531)
- Transaction frequency: 17/month

**Insight:** 51.67% of months classified as "Big Spender" behavior

---

#### **Model Performance Summary**
| Model | Type | Performance | Best Use Case |
|-------|------|-------------|---------------|
| Random Forest | Regression | R² = 0.48 | Transaction prediction |
| XGBoost | Regression | R² = 0.41 | Feature importance analysis |
| Isolation Forest | Anomaly | 5.07% detected | Fraud detection |
| K-Means | Clustering | 2 personas | Personalization |
| ARIMA | Time Series | RMSE = $5,263 | Long-term forecasting |

---

## Visualizations

### Dashboard Preview

#### 1. Financial Overview
![Income vs Expense Timeline](visualizations/income_vs_expense.png)
*Monthly income and expense trends over 5-year period*

#### 2. Model Performance Comparison
![Model Comparison](visualizations/model_comparison.png)
*Comparative analysis of all 5 ML models (R² scores)*

#### 3. Anomaly Detection
![Anomaly Scatter](visualizations/anomaly_detection.png)
*Unusual transactions highlighted in red (62 anomalies detected)*

#### 4. Spending Personas
![Personas PCA](visualizations/spending_personas.png)
*K-Means clustering showing Big Spender vs Frugal Saver personas*

#### 5. 6-Month Forecast
![Forecast](visualizations/forecast_chart.png)
*ARIMA forecast with 95% confidence intervals*

#### 6. Category Breakdown
![Category Pie](visualizations/category_breakdown.png)
*Spending distribution across 10 categories*

> **Note:** All visualizations are interactive Plotly charts with hover details, zoom, and export capabilities

---

## Installation

### Prerequisites
```bash
Python 3.9 or higher
pip (Python package manager)
Git
```

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/personal-finance-ml-project.git
cd personal-finance-ml-project
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import pandas, sklearn, xgboost, plotly; print('All packages installed successfully!')"
```

---

## Usage

### Quick Start

#### 1. Run Data Cleaning
```bash
python scripts/data_cleaning.py
```
**Output:** `Personal_Finance_CLEANED.csv` with 27 features

---

#### 2. Create SQL Database
```bash
python sql/create_database.py
```
**Output:** `personal_finance.db` (560 KB)

---

#### 3. Train ML Models
```bash
python scripts/train_models.py
```
**Output:** 5 trained models saved in `models/` folder

---

#### 4. Generate Visualizations
```bash
jupyter notebook notebooks/03_Phase3_Visualizations.ipynb
```
**Output:** 50+ interactive charts

---

### Using the SQL Database
```python
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('sql/personal_finance.db')

# Example: Get monthly spending by category
query = """
SELECT category, SUM(amount) as total
FROM transactions
WHERE type = 'Expense'
GROUP BY category
ORDER BY total DESC
"""

df = pd.read_sql_query(query, conn)
print(df)
```

**See `sql/sql_queries.sql` for 55+ pre-written queries**

---

### Using Pre-trained Models
```python
import pickle
import pandas as pd

# Load Random Forest model
with open('models/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Make predictions
predictions = rf_model.predict(X_test)
```

---

##  Project Structure
```
personal-finance-ml-project/
│
├── 📄 README.md                    # You are here
├── 📄 requirements.txt             # Python dependencies
├── 📄 .gitignore                   # Git ignore rules
│
├── 📂 data/
│   ├── Personal_Finance_CLEANED.csv       # Clean dataset (27 features)
│   └── data_dictionary.md                 # Feature descriptions
│
├── 📂 sql/
│   ├── personal_finance.db               # SQLite database (560 KB)
│   ├── sql_queries.sql                   # 55+ analytical queries
│   ├── create_database.py                # Database builder script
│   └── SQL_README.md                     # SQL documentation
│
├── 📂 notebooks/
│   ├── 01_Phase1_Data_Cleaning.ipynb    # Data preparation
│   ├── 02_Phase2_Machine_Learning.ipynb  # Model training
│   └── 03_Phase3_Visualizations.ipynb    # Dashboard creation
│
├── 📂 models/
│   ├── xgb_model.pkl                     # XGBoost model
│   ├── rf_model.pkl                      # Random Forest (BEST)
│   ├── isolation_forest_model.pkl        # Anomaly detector
│   ├── kmeans_model.pkl                  # Clustering model
│   └── arima_model.pkl                   # Time series model
│
├── 📂 results/
│   ├── xgb_predictions.csv               # XGBoost predictions
│   ├── rf_predictions.csv                # Random Forest predictions
│   ├── model_comparison.csv              # Model metrics
│   ├── anomaly_detection_results.csv     # Flagged transactions
│   ├── spending_personas.csv             # Cluster assignments
│   └── arima_forecast_6months.csv        # Future predictions
│
├── 📂 visualizations/
│   ├── income_vs_expense.png             # Timeline chart
│   ├── model_comparison.png              # Model performance
│   ├── anomaly_detection.png             # Scatter plot
│   ├── spending_personas.png             # PCA clustering
│   ├── forecast_chart.png                # ARIMA forecast
│   └── category_breakdown.png            # Pie chart
│
├── 📂 scripts/
│   ├── data_cleaning.py                  # Data preprocessing
│   ├── feature_engineering.py            # Feature creation
│   └── train_models.py                   # Model training pipeline
│
├── 📂 docs/
│   ├── PROJECT_REPORT.md                 # Detailed findings
│   ├── METHODOLOGY.md                    # Technical approach
│   └── RESULTS_SUMMARY.md                # Executive summary
│
└── 📂 excel/
    └── Personal_Finance_Analysis.xlsx    # Excel workbook
```

---

## Skills Demonstrated

### **Data Science & ML**
✅ Data Cleaning & Preprocessing  
✅ Feature Engineering (27 features)  
✅ Supervised Learning (Regression)  
✅ Unsupervised Learning (Clustering, Anomaly Detection)  
✅ Time Series Forecasting (ARIMA)  
✅ Model Evaluation & Selection  
✅ Hyperparameter Tuning  

### **Programming & Tools**
✅ Python (pandas, NumPy, scikit-learn, XGBoost)  
✅ SQL (Database Design, Query Optimization)  
✅ Jupyter Notebooks  
✅ Git/GitHub  
✅ Google Colab  

### **Data Visualization**
✅ Plotly (Interactive Dashboards)  
✅ Matplotlib & Seaborn  
✅ Business Intelligence Reporting  

### **Business Skills**
✅ Problem Definition  
✅ Insight Generation  
✅ Stakeholder Communication  
✅ Recommendation Development  

---

##  Future Enhancements

### **Short-term (Next 2-4 weeks)**
- [ ] Deploy Streamlit dashboard to cloud (Heroku/Streamlit Cloud)
- [ ] Add automated PDF report generation
- [ ] Implement email alert system for anomalies
- [ ] Create REST API for model predictions

### **Medium-term (1-3 months)**
- [ ] Add deep learning model (LSTM for time series)
- [ ] Implement ensemble model combining top performers
- [ ] Build mobile-responsive dashboard
- [ ] Add multi-currency support

### **Long-term (3-6 months)**
- [ ] Real-time transaction processing pipeline
- [ ] Integration with banking APIs (Plaid)
- [ ] Natural Language Processing for transaction categorization
- [ ] Recommendation engine for budget optimization
- [ ] Multi-user support with authentication

---

##  Documentation

- **[Project Report](docs/PROJECT_REPORT.md)** - Detailed analysis and findings
- **[Methodology](docs/METHODOLOGY.md)** - Technical approach and decisions
- **[Results Summary](docs/RESULTS_SUMMARY.md)** - Executive overview
- **[SQL Guide](sql/SQL_README.md)** - Database documentation
- **[Data Dictionary](data/data_dictionary.md)** - Feature descriptions

---

##  Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Author

**Jitesh Yadav**  
Researcher

[![LinkedIn](https://www.linkedin.com/in/jitesh-yadav-9a97b1165/)
[![Email](jitesh3777yadavv@gmail.com)

---

## Acknowledgments

- Dataset inspired by real-world personal finance tracking
- Machine learning techniques based on industry best practices
- Visualization design influenced by modern BI dashboards
- Project structure follows data science project templates

---

### If you found this project helpful, please consider giving it a star!

**Built with ❤️ and ☕ by Jitesh **
