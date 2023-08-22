#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 05:06:22 2023

@author: Nunoo Emmanuel Felix Landlord
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

# import dataset
 df = pd.read_csv("/Users/mensahotooemmanuel/Movies/FS_spyder/FScleaned.csv")

df.columns

df_model = df[['site', 'hhedu', 'hhhage', 'hhhsex', 'hhethnic', 'hhsize', 'u05',
       'povline', 'windex3', 'windex5', 'FS_score', 'FS']]

# get dummy data 
df_dum = pd.get_dummies(df_model)


# train test split 
from sklearn.model_selection import train_test_split

df_dum.columns

X = df_dum.drop(['FS_score'], axis =1)
y = df_dum.FS_score.values

# set train 80% and test 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# machine learning models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# Decision Tree builing
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)


# Random Forest building
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Gradient Boosting building
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)

# evaluating each model
def evaluate_model(model_name, y_true, y_pred):
    print(f"--- {model_name} ---")
    print(classification_report(y_true, y_pred, zero_division= 1))
    print("=" * 50)

evaluate_model("Decision Tree", y_test, dt_predictions)
evaluate_model("Random Forest", y_test, rf_predictions)
evaluate_model("Gradient Boosting", y_test, gb_predictions)


# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Estimator:", grid_search.best_estimator_)

best_rf_model = grid_search.best_estimator_
best_rf_predictions = best_rf_model.predict(X_test)

evaluate_model("Best Random Forest", y_test, best_rf_predictions)



# feature importance for random forest

feature_names = df.columns

# Print feature names
print(feature_names)

feature_importances = rf_model.feature_importances_

# Pair feature names with their importance scores
importances = dict(zip(feature_names, feature_importances))

# Print feature importances
for feature, importance in importances.items():
    print(f"{feature}: {importance}")



# feature importance for decision trees

feature_names = df.columns

# Print feature names
print(feature_names)

feature_importances = dt_model_model.feature_importances_

# Pair feature names with their importance scores
importances = dict(zip(feature_names, feature_importances))

# Print feature importances
for feature, importance in importances.items():
    print(f"{feature}: {importance}")



# feature importance for gradient boosting

feature_names = df.columns

# Print feature names
print(feature_names)

feature_importances = gb_model.feature_importances_

# Pair feature names with their importance scores
importances = dict(zip(feature_names, feature_importances))

# Print feature importances
for feature, importance in importances.items():
    print(f"{feature}: {importance}")

