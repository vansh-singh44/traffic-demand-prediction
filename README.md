# 🚦 Traffic Demand Prediction

## Gridlock Hackathon 2.0

An end-to-end Machine Learning solution developed for **Gridlock Hackathon 2.0** to predict urban traffic demand using historical traffic patterns, weather conditions, road characteristics, and location-based information.

---

## 📌 Project Objective

The goal of this project is to accurately forecast urban traffic demand using advanced feature engineering and ensemble machine learning techniques.

---

## 🔍 Methodology

### Data Preprocessing
- Missing value handling
- Data cleaning and formatting
- Log transformation of the target variable

### Feature Engineering
- Time-based features
- Weather-based features
- Road and lane information
- Geohash decomposition
- Interaction features
- Cyclic encoding using sine and cosine transformations

### Encoding Strategy
- K-Fold Target Encoding for categorical variables

### Machine Learning Models
- LightGBM Regressor
- CatBoost Regressor
- XGBoost Regressor

### Validation Strategy
- 10-Fold Cross Validation
- Early Stopping

### Ensemble Learning

Final predictions are generated using a weighted ensemble of three gradient boosting models:

- LightGBM : 30%
- CatBoost : 45%
- XGBoost : 25%

The ensemble approach helps reduce overfitting and improves the robustness of predictions.

---

## 📂 Project Structure

```
traffic-demand-prediction/
│
├── traffic_demand_prediction.py
├── Submission_blend.csv
├── Requirements.txt
└── README.md
```

---

## 📦 Libraries Used

- pandas
- numpy
- scikit-learn
- lightgbm
- catboost
- xgboost

---

## ▶️ Running the Project

Install dependencies:

```bash
pip install -r Requirements.txt
```

Run the model:

```bash
python traffic_demand_prediction.py
```

The script automatically generates:

```
Submission_blend.csv
```

---

## 🏆 Competition

**Gridlock Hackathon 2.0**

**Problem Statement:** Traffic Demand Prediction for Urban Mobility Analytics

---

## 👨‍💻 Author

**Vansh Singh**

GitHub: https://github.com/vansh-singh44

---

## 📄 License

This project is shared for educational and portfolio purposes.
