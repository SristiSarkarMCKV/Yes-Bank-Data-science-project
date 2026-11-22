# 📈 Yes Bank Stock Closing Price Prediction

A machine learning regression project that predicts the monthly closing stock price (`Close`) of **Yes Bank** using historical monthly OHLC (`Open`, `High`, `Low`, `Close`) records from July 2005 to November 2020. The project evaluates how linear, regularized linear, and tree-based ensemble models perform across both steady bull-market growth and extreme structural shocks—most notably the **September 2018 Rana Kapoor fraud and governance crisis**.


## 📌 Table of Contents

- [Business Context & Objective](#-business-context--objective)
- [Dataset Overview](#-dataset-overview)
- [Project Architecture](#-project-architecture)
- [Exploratory Data Analysis & Key Insights](#-exploratory-data-analysis--key-insights)
- [Model Comparison & Results](#-model-comparison--results)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Conclusion & Future Scope](#-conclusion--future-scope)


## 🎯 Business Context & Objective

Yes Bank is a prominent Indian private-sector bank whose stock price experienced rapid multi-year growth until mid-2018, followed by a steep collapse triggered by regulatory intervention and the exit of CEO Rana Kapoor in September 2018.

- **Primary Objective:** Build and evaluate regression models capable of accurately predicting Yes Bank's monthly closing stock price (`Close`) from intra-month trading boundaries (`Open`, `High`, `Low`).
- **Key Question:** Can standard regression and regularized linear models maintain high predictive accuracy across dramatic regime shifts and sharp market downturns?

## 📊 Dataset Overview

The dataset (`data_YesBank_StockPrices.csv`) contains **185 monthly records** spanning **July 2005 – November 2020** with zero missing or duplicate entries.

| Feature | Description | Data Type |
| :--- | :--- | :--- |
| `Date` | Month and year of the trading record (`Mon-YY`) | Object / Datetime |
| `Open` | Opening stock price for the month (INR) | Float |
| `High` | Highest stock price recorded during the month (INR) | Float |
| `Low` | Lowest stock price recorded during the month (INR) | Float |
| `Close` | **Target Variable:** Closing stock price at month-end (INR) | Float |

- **Source URL:** `https://raw.githubusercontent.com/SristiSarkarMCKV/Yes-Bank-Stock-Closing-Price-Prediction/main/data_YesBank_StockPrices.csv`


## 🏗️ Project Architecture

1. **Data Loading & Clean-Up:** Stream raw CSV data, parse `Date` into chronological datetime objects, sort records ascending, and verify zero null/duplicate entries.
2. **Exploratory Data Analysis (EDA):**
   - Analyze historical trends using 6-month and 12-month rolling moving averages.
   - Validate the hypothesis that $\text{Close}_t$ strongly correlates with $\text{Open}_{t+1}$ ($r = 0.9999$).
   - Inspect price distributions via boxplots (retaining genuine 2017–2018 peak outliers).
   - Quantify multicollinearity using **Variance Inflation Factor (VIF)**.
3. **Preprocessing & Chronological Split:**
   - Split data chronologically (**80% Train:** Jul 2005 – Oct 2017 | **20% Test:** Nov 2017 – Nov 2020) to prevent look-ahead data leakage.
   - Standardize features using `StandardScaler` fit strictly on the training set.
4. **Model Training & Hyperparameter Tuning:**
   - Experiment across 6 regression algorithms (`Linear Regression`, `Ridge`, `Lasso`, `ElasticNet`, `Random Forest`, `Gradient Boosting`).
   - Tune hyperparameters via `GridSearchCV` with `TimeSeriesSplit(n_splits=5)`.
5. **Evaluation & Explainability:** Compare models using `MSE`, `RMSE`, `MAE`, `MAPE`, and $R^2$, and examine standardized feature coefficients.


## 🔍 Exploratory Data Analysis & Key Insights

- **2018 Structural Break:** Stock prices climbed steadily from ~₹12 in 2005 to a peak of **₹367.90** in July 2018 before plummeting below ₹20 in 2020.
- **High Multicollinearity:** `Open` ($\text{VIF} = 175.19$), `High` ($\text{VIF} = 167.06$), and `Low` ($\text{VIF} = 71.57$) exhibit pairwise correlations above $0.978$, making L1/L2 regularization critical for coefficient stability.
- **Outlier Retention:** Upper-tail IQR outliers correspond to valid historical highs during 2017–2018 and are preserved so the model learns high-valuation regimes.


## 🏆 Model Comparison & Results

Evaluated on the out-of-time test set (**Nov 2017 – Nov 2020**, covering the 2018 peak and crash):

| Model | Best Hyperparameters | MSE | RMSE (₹) | MAE (₹) | MAPE (%) | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ElasticNet** | `alpha=0.001, l1_ratio=0.9` | **249.12** | **15.78** | 9.86 | 11.94% | **0.9848** |
| **Lasso Regression** | `alpha=0.001` | 249.99 | 15.81 | 9.79 | **11.88%** | **0.9848** |
| **Ridge Regression** | `alpha=0.001` | 250.25 | 15.82 | 9.79 | **11.88%** | **0.9848** |
| **Linear Regression** | Default | 250.33 | 15.82 | **9.78** | **11.88%** | **0.9848** |
| **Random Forest** | `max_depth=5, n_estimators=200` | 472.30 | 21.73 | 14.06 | 16.25% | 0.9712 |
| **Gradient Boosting** | `lr=0.1, max_depth=3, n_estimators=100` | 595.31 | 24.40 | 13.56 | 14.90% | 0.9637 |

### Key Findings

- **Regularized Linear Models Win:** **ElasticNet / Lasso Regression** achieves an $R^2$ **of 0.9848** and an RMSE of **₹15.78**, outperforming tree ensembles because linear combinations of `Open`, `High`, and `Low` extrapolate smoothly across sharp market drops.
- **Feature Importance (Standardized Coefficients):** `High` (`+73.51`) and `Low` (`+63.31`) act as strong positive anchors, while `Open` (`-52.70`) captures intra-month directional reversal.


## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning & Statistics:** `scikit-learn`, `statsmodels`


## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/SristiSarkarMCKV/Yes-Bank-Stock-Closing-Price-Prediction.git
cd Yes-Bank-Stock-Closing-Price-Prediction
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

### 3. Run the Pipeline

Run the Python script or Jupyter/Colab notebook—the data loader automatically fetches `data_YesBank_StockPrices.csv` directly from GitHub:

```bash
python yes_bank_stock_prediction.py
```


## 💡 Conclusion & Future Scope

- **Conclusion:** Despite extreme volatility following the 2018 governance crisis, regularized linear models (**ElasticNet** and **Lasso**) effectively handle severe OHLC multicollinearity and accurately estimate monthly closing prices ($R^2 = 0.9848$).
- **Future Scope:**
  - Engineer lagged features ($\mathrm{Close}_{t-1}, \mathrm{Close}_{t-2}$) and rolling volatility indicators for multi-step forward forecasting without intra-month `High`/`Low` inputs.
  - Incorporate macroeconomic indicators and financial news sentiment analysis to capture sudden corporate governance shocks earlier.
