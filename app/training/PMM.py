# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import lightgbm as lgb
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

# 2. Load dataset
df = pd.read_csv('../../data/predictive_maintenance.csv')



# 3. Check data
print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# 4. Check missing values
print(df.isnull().sum())

# 5. Check distribution of target (failure)
print(df['failure'].value_counts())

# 6. Drop unnecessary columns
columns_to_drop = ['machine_id', 'timestamp']  # Update based on your dataset
df = df.drop(columns=columns_to_drop, axis=1, errors='ignore')

# 7. Separate features and target
X = df.drop('failure', axis=1)
y = df['failure']

# 8. Drop non-numeric columns
non_numeric_columns = X.select_dtypes(include=['object']).columns
print("Dropping non-numeric columns:", non_numeric_columns.tolist())
X = X.drop(non_numeric_columns, axis=1)

# 9. Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("Scaled Features Example:")
print(X_scaled[:5])

# 10. Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print(f"Original dataset shape: {X_scaled.shape}, {y.shape}")
print(f"After SMOTE: {X_resampled.shape}, {y_resampled.shape}")

# 11. Train-Test Split AFTER SMOTE (IMPORTANT!)
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled,
    test_size=0.2,
    random_state=42,
    stratify=y_resampled
)

print("Training Set:", X_train.shape, y_train.shape)
print("Test Set:", X_test.shape, y_test.shape)

# 12. Hyperparameter Tuning using GridSearchCV
lgb_model = lgb.LGBMClassifier(objective='binary', is_unbalance=True, random_state=42)

param_grid = {
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200],
    'num_leaves': [31, 63],
    'max_depth': [-1, 5],
}


grid = GridSearchCV(
    estimator=lgb_model,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=2,
    n_jobs=4  

)

# Fit the model
grid.fit(X_train, y_train)

# Best Hyperparameters
print("Best Hyperparameters:")
print(grid.best_params_)

# Best model
best_lgb_model = grid.best_estimator_

# Evaluate
y_pred_best = best_lgb_model.predict(X_test)
print("\nConfusion Matrix (Best LightGBM):")
print(confusion_matrix(y_test, y_pred_best))

print("\nClassification Report (Best LightGBM):")
print(classification_report(y_test, y_pred_best))

print(f"Best LightGBM Accuracy: {accuracy_score(y_test, y_pred_best)*100:.2f}%")

# Save Best Model
joblib.dump(best_lgb_model, 'lightgbm_best_model.pkl')
print("\n Best Model Saved as 'lightgbm_best_model.pkl'!")

# 13. Random Forest Classifier (after SMOTE)
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

# Confusion Matrix
print("\nConfusion Matrix (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))

# Classification Report
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)*100:.2f}%")

# Predict Probabilities
y_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Sample Probabilities
for i in range(10):
    print(f"Machine {i+1}: Failure Probability = {y_pred_prob_rf[i]*100:.2f}%")

# Plot Distribution
plt.figure(figsize=(8,5))
plt.hist(y_pred_prob_rf, bins=50, color='blue', edgecolor='black')
plt.title('Random Forest: Distribution of Failure Probabilities')
plt.xlabel('Failure Probability')
plt.ylabel('Number of Machines')
plt.savefig('random_forest_probability_distribution.png')
plt.close()

# 14. LightGBM Model again (final)
lgb_model_final = lgb.LGBMClassifier(
    objective='binary',
    is_unbalance=True,
    random_state=42,
    n_estimators=100
)

lgb_model_final.fit(X_train, y_train)

y_pred_lgb = lgb_model_final.predict(X_test)
y_pred_prob_lgb = lgb_model_final.predict_proba(X_test)[:, 1]

# LightGBM evaluation
print("\nConfusion Matrix (LightGBM Final):")
print(confusion_matrix(y_test, y_pred_lgb))

print("\nClassification Report (LightGBM Final):")
print(classification_report(y_test, y_pred_lgb))

print(f"LightGBM Final Accuracy: {accuracy_score(y_test, y_pred_lgb)*100:.2f}%")

# Threshold tuning
threshold = 0.3
y_pred_lgb_tuned = (y_pred_prob_lgb >= threshold).astype(int)

print("\nConfusion Matrix (LightGBM Tuned Threshold):")
print(confusion_matrix(y_test, y_pred_lgb_tuned))

print("\nClassification Report (LightGBM Tuned Threshold):")
print(classification_report(y_test, y_pred_lgb_tuned))

print(f"LightGBM Accuracy after Threshold Tuning: {accuracy_score(y_test, y_pred_lgb_tuned)*100:.2f}%")

# Save final model
joblib.dump(lgb_model_final, 'lightgbm_predictive_maintenance_model.pkl')
print("\n LightGBM Model saved as 'lightgbm_predictive_maintenance_model.pkl'!")

# 15. Feature Importance
importance = lgb_model_final.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10,6))
plt.barh(feature_names, importance)
plt.title('Feature Importance in Predictive Maintenance')
plt.xlabel('Importance Score')
plt.ylabel('Sensor Features')
plt.show()
