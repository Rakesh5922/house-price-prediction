# 🏠 House Price Prediction — End-to-End ML Project

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR_APP_URL.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

> Predicts house sale prices in Ames, Iowa using a tuned Gradient Boosting model.  
> Deployed as an interactive web app via Streamlit Cloud.

---

## 🔗 Live Demo
👉 **[Click here to try the app](https://YOUR_APP_URL.streamlit.app)**

> 🖼️ **App Screenshot:**  
> ![App Screenshot](images/app_screenshot.png)  
> *(Take a screenshot of your running Streamlit app and save it as `images/app_screenshot.png`)*

---

## 📌 Project Overview

This is a complete, production-style machine learning project covering every stage of the data science workflow — from raw data to a deployed web application.

| Stage | What Was Done |
|-------|--------------|
| 📊 EDA | Analyzed 80+ features, distributions, correlations, outliers |
| 🧹 Data Cleaning | Handled 19+ missing value columns using domain knowledge |
| ⚙️ Feature Engineering | Created 11 new features (TotalSF, HouseAge, QualXArea, etc.) |
| 🔢 Encoding | Ordinal encoding for quality ratings + One-hot for categories |
| 🤖 Modeling | Compared Ridge, Random Forest, and Gradient Boosting |
| 🎯 Tuning | RandomizedSearchCV with 5-Fold Cross Validation |
| 📈 Evaluation | R², RMSE, MAPE, Residual Analysis |
| 🚀 Deployment | Streamlit web app with live predictions |

---

## 📊 Model Performance

| Model | R² Score | RMSE |
|-------|----------|------|
| Ridge Regression (baseline) | ~0.87 | ~$20,245 |
| Random Forest | ~0.89 | ~$19,800 |
| **Gradient Boosting (tuned)** ✅ | **~0.91** | **~$18,500** |

> Final model predicts house prices within **±13% accuracy**

---

## 🗂️ All 13 Lessons — Complete Walkthrough

---

### 📘 Lesson 1 — Data Loading & Exploration
Load the Ames Housing dataset and understand its structure — shape, data types, and basic statistics.

```python
train = pd.read_csv('train.csv')
print("Train shape:", train.shape)   # (1460, 81)
train.describe()
```

---

### 📘 Lesson 2 — Target Variable Analysis
Visualize SalePrice distribution — it is right-skewed, so we apply log transformation to make it normal.

| Before Log Transform | After Log Transform |
|---------------------|---------------------|
| ![SalePrice Distribution](images/06_saleprice_distribution.png) | ![Log SalePrice](images/07_log_saleprice_distribution.png) |

> 💡 **Why?** Models perform better when the target is normally distributed.

---

### 📘 Lesson 3 — Correlation Analysis
Find which features are most strongly correlated with SalePrice.

![Top Correlations](images/08_top_correlations.png)

![Correlation Heatmap](images/09_correlation_heatmap.png)

> 💡 Top features: `OverallQual`, `GrLivArea`, `TotalBsmtSF`, `GarageCars`

---

### 📘 Lesson 4 — Exploratory Data Analysis (EDA)
Visualize key relationships between features and SalePrice.

**Overall Quality vs Price:**  
![Quality vs Price](images/10_quality_vs_price.png)

**Living Area vs Price:**  
![Living Area vs Price](images/11_livingarea_vs_price.png)

**Neighborhood vs Price:**  
![Neighborhood vs Price](images/12_neighborhood_vs_price.png)

---

### 📘 Lesson 5 — Outlier Detection & Removal
Remove houses with very large area but suspiciously low price — these are data errors.

```python
# Before: 1460 rows
train = train[~((train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000))]
# After: 1458 rows — 2 outliers removed
```

---

### 📘 Lesson 6 — Missing Value Treatment
Handle 19+ columns with missing data using domain knowledge (not just filling with mean).

| Strategy | Columns | Reason |
|----------|---------|--------|
| Fill `"None"` | PoolQC, Fence, Alley, FireplaceQu... | Missing = feature doesn't exist |
| Fill `0` | GarageArea, BsmtFinSF1, MasVnrArea... | Missing = 0 sqft |
| Fill neighborhood median | LotFrontage | Similar houses have similar lot sizes |
| Fill mode | Electrical | Only 1 missing value |

```python
print("Missing values remaining:", train.isnull().sum().sum())  # → 0
```

---

### 📘 Lesson 7 — Feature Engineering
Create 11 new features that capture information better than the original columns.

| New Feature | Formula | Why It Helps |
|-------------|---------|--------------|
| `TotalSF` | Basement + 1stFlr + 2ndFlr | Total size is more important than individual floors |
| `TotalBaths` | FullBath + BsmtFullBath | Combined bathroom count |
| `HouseAge` | YrSold - YearBuilt | Age at time of sale matters more than build year |
| `IsRemodeled` | 1 if remodeled else 0 | Binary flag adds signal |
| `HasGarage` | 1 if GarageArea > 0 | Presence/absence is a clear signal |
| `HasFireplace` | 1 if Fireplaces > 0 | Presence/absence is a clear signal |
| `QualXArea` | OverallQual × TotalSF | Interaction: quality AND size together |

**Before vs After Feature Engineering:**  
![Feature Engineering](images/23_feature_engineering_comparison.png)

---

### 📘 Lesson 8 — Encoding Categorical Variables
Convert text categories to numbers so the model can process them.

- **Ordinal Encoding** — for quality ratings (Po=1, Fa=2, TA=3, Gd=4, Ex=5)
- **One-Hot Encoding** — for all other text columns (Neighborhood, HouseStyle, etc.)

```python
# Columns before encoding: ~85
# Columns after encoding:  ~220+
train = pd.get_dummies(train, drop_first=True)
```

![Exterior Quality Encoded](images/28_exterior_quality_encoded.png)

---

### 📘 Lesson 9 — Model Building & Comparison
Train and compare three models.

![Model Comparison](images/36_model_comparison.png)

| Model | Key Insight |
|-------|------------|
| **Ridge Regression** | Fast, good baseline — R² ~0.87 |
| **Random Forest** | Better — handles non-linearity — R² ~0.89 |
| **Gradient Boosting** | Best — learns from errors sequentially — R² ~0.91 |

**Ridge Actual vs Predicted:**  
![Ridge Predictions](images/32_ridge_actual_vs_predicted.png)

---

### 📘 Lesson 10 — Cross Validation & Hyperparameter Tuning
Use 5-Fold CV to get reliable scores, then tune with RandomizedSearchCV.

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(gbm, X_train, y_train, cv=kf, scoring='r2')
print(f"Mean CV R²: {cv_scores.mean():.4f}")
```

**RMSE Before vs After Tuning:**  
![RMSE Comparison](images/41_rmse_comparison.png)

---

### 📘 Lesson 11 — Model Evaluation & Residual Analysis
Deeply evaluate the final tuned model beyond just R².

**Top 15 Most Important Features:**  
![Feature Importance](images/37_feature_importance.png)

**Residuals vs Predicted:**  
![Residuals vs Predicted](images/43_residuals_vs_predicted.png)

**Residuals Distribution (bell-shaped = good):**  
![Residuals Distribution](images/44_residuals_distribution.png)

**Final Actual vs Predicted:**  
![Final Predictions](images/45_final_actual_vs_predicted.png)

```
========================================
   FINAL MODEL SCORECARD
========================================
  Model    : Tuned Gradient Boosting
  R²       : ~0.913
  RMSE     : ~$18,500
  MAPE     : ~9.2%
  Accuracy : Predicts within ±13%
========================================
```

---

### 📘 Lesson 12 — Model Saving
Save the trained model and feature names so the Streamlit app can load them.

```python
import joblib, os
os.makedirs('models', exist_ok=True)
joblib.dump(best_model,               'models/house_price_model.pkl')
joblib.dump(X_train.columns.tolist(), 'models/feature_names.pkl')
print("Model saved!")
```

---

### 📘 Lesson 13 — Streamlit App & Deployment
Built a full interactive web app with real-time predictions.

**Features:**
- 🎛️ Sliders for all house features
- 💰 Real-time price prediction
- 📊 Confidence range (±13%)
- 📋 Full input summary table

**Deployed on Streamlit Cloud:**  
👉 [YOUR_APP_URL.streamlit.app](https://YOUR_APP_URL.streamlit.app)

> 🖼️ Add your app screenshot here:  
> ![Streamlit App](images/app_screenshot.png)

---

## 🛠️ Tech Stack

- **Python** — pandas, numpy, scikit-learn, matplotlib, seaborn
- **ML** — GradientBoostingRegressor, RandomizedSearchCV, KFold CV
- **App** — Streamlit, joblib
- **Data** — [Ames Housing Dataset — Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

---

## 📁 Project Structure

```
house-price-prediction/
│
├── app.py                        # Streamlit web application
├── house-price-prediction.ipynb  # Full ML notebook
├── requirements.txt              # Python dependencies
├── models/
│   ├── house_price_model.pkl     # Trained GBM model
│   └── feature_names.pkl         # Feature list for inference
├── images/                       # All chart screenshots
├── train.csv                     # Training data
└── test.csv                      # Test data
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 What I Learned

- How to handle real-world missing data using **domain knowledge**
- How **feature engineering** improves performance more than algorithm choice
- Why **log-transforming** skewed targets leads to better predictions
- How to use **cross-validation** to get reliable model scores
- How to deploy an ML model as a **production web app**

---

## 👤 Author

**Rakesh**  
📧 your-email@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)  
🐙 [GitHub](https://github.com/YOUR_USERNAME)

---

## 📄 License
This project is open source under the [MIT License](LICENSE).
