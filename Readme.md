GRIDLOCK HACKATHON 2.0

Project:
Traffic Demand Prediction for Urban Mobility Analytics

Objective:
Develop a machine learning model to predict travel demand patterns using historical traffic, weather, road, and location information.

Approach:
1. Data preprocessing and missing value handling.
2. Log transformation of target variable.
3. Feature engineering based on:
   - Time-based features
   - Weather features
   - Road information
   - Geohash encoding
   - Interaction features
4. Target Encoding using K-Fold strategy.
5. Model training using:
   - LightGBM
   - CatBoost
   - XGBoost
6. 10-Fold Cross Validation.
7. Weighted Ensemble:
      LightGBM : 30%
      CatBoost : 45%
      XGBoost : 25%
8. Final predictions generated in submission_blend.csv.

Libraries Used:
- pandas
- numpy
- scikit-learn
- lightgbm
- catboost
- xgboost

Author:
Vansh Singh

Submission:
Gridlock Hackathon 2.0