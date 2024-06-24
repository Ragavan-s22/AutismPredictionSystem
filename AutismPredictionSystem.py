import pandas as pd
import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

# Initialize label encoder
le = LabelEncoder()

# Load data
train_df = pd.read_csv('Autism/train.csv')
test_df = pd.read_csv('Autism/test.csv')

# Data preprocessing
train_df.loc[train_df["ethnicity"].isin({"Pasifika", "Hispanic", "Turkish"}), "ethnicity"] = "Others"
test_df.loc[test_df["ethnicity"].isin({"Pasifika", "Hispanic", "Turkish"}), "ethnicity"] = "Others"

train_df['austim'] = le.fit_transform(train_df['austim'])
test_df['austim'] = le.transform(test_df['austim'])
train_df['ethnicity'] = le.fit_transform(train_df['ethnicity'])
test_df['ethnicity'] = le.transform(test_df['ethnicity'])

# Define feature and target variables
X = train_df[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
              'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'austim', 'result', 'ethnicity']]
y = train_df['Class/ASD']
X_test = test_df[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                  'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'austim', 'result', 'ethnicity']]

# Function to perform Grid Search CV and get predictions
def perform_grid_search_cv(model, param_grid, X, y, X_test):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    grid_model = GridSearchCV(model, param_grid, cv=kf, n_jobs=-1)  # Use all CPU cores
    grid_model.fit(X, y)
    return grid_model.predict(X_test), grid_model

# Define smaller parameter grids for each model
param_grid_gb = {'n_estimators': [50, 150, 200],
                 'max_depth': [2, 4, 6]}
param_grid_rf = {'n_estimators': [50, 150, 200],
                 'max_depth': [2, 4, 6]}
param_grid_ad = {'n_estimators': [50, 150, 200]}
param_grid_b = {'n_estimators': [50, 150, 200]}
param_grid_xt = {'n_estimators': [50, 150, 200],
                 'max_depth': [2, 4, 6]}
param_grid_lr = {"C": np.logspace(-3, 3, 5), "penalty": ["l1", "l2"]}

# Perform grid search CV for each model
preds_gb, gb_model = perform_grid_search_cv(GradientBoostingClassifier(random_state=0), param_grid_gb, X, y, X_test)
preds_rf, rf_model = perform_grid_search_cv(RandomForestClassifier(random_state=0), param_grid_rf, X, y, X_test)
preds_ad, ad_model = perform_grid_search_cv(AdaBoostClassifier(random_state=0, algorithm='SAMME'), param_grid_ad, X, y, X_test)
preds_b, b_model = perform_grid_search_cv(BaggingClassifier(random_state=0), param_grid_b, X, y, X_test)
preds_xt, xt_model = perform_grid_search_cv(ExtraTreesClassifier(random_state=0), param_grid_xt, X, y, X_test)

# Logistic Regression
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
grid_model_lr = GridSearchCV(LogisticRegression(solver='saga', tol=1e-5, max_iter=10000, random_state=0), param_grid_lr, cv=kf, n_jobs=-1)
grid_model_lr.fit(X, y)
preds_lr = grid_model_lr.predict(X_test)

# Combining predictions from different models (here we use the majority voting approach)
final_preds = (preds_gb + preds_rf + preds_ad + preds_b + preds_xt + preds_lr) / 6
final_preds = np.where(final_preds > 0.5, 1, 0)  # Convert probabilities to binary predictions

# Print individuals predicted to have autism
autism_predictions = test_df.loc[final_preds == 1]
print("Individuals predicted to have autism:")
print(autism_predictions)

# Add a new column to the test dataframe indicating "yes" or "no" for autism
test_df['Autism_Prediction'] = np.where(final_preds == 1, 'Yes', 'No')

# Save the final predictions to CSV
submission = pd.read_csv('Autism/sample_submission.csv')
submission['Class/ASD'] = final_preds
submission['Autism_Prediction'] = test_df['Autism_Prediction']
submission.to_csv('final_predictions.csv', index=False)

# Print the updated test dataframe with the new column
print(test_df.head())