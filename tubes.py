import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('bank.csv')

# Preprocessing
df = df.drop_duplicates()
categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    df[column] = df[column].astype('category')
df_encoded = pd.get_dummies(df, drop_first=True)

# Define target and features
target = 'deposit_yes'  # Adjust this to the actual target column name
features = df_encoded.drop(columns=[target])

X_train, X_test, y_train, y_test = train_test_split(features, df_encoded[target], test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training SVM model with GridSearchCV...")
svm_model = SVC()
param_grid_svm = {
    'C': [10, 100, 1000],
    'gamma': [0.01, 0.001],
    'kernel': ['rbf']
}
grid_search_svm = GridSearchCV(estimator=svm_model, param_grid=param_grid_svm, cv=5, n_jobs=-1, verbose=2)
grid_search_svm.fit(X_train_scaled, y_train)

# Get the best model
best_svm_model = grid_search_svm.best_estimator_

# Save the model
joblib.dump(best_svm_model, 'svm_model.joblib')

loaded_model = joblib.load('svm_model.joblib')
print("Model yang dimuat:")
print(loaded_model)

# Evaluate the loaded model
result_loaded = loaded_model.score(X_test_scaled, y_test)
print(f'Hasil model menggunakan model yang sudah disimpan: {result_loaded}')

# Predict using the loaded model
y_pred_svm = loaded_model.predict(X_test_scaled)

# Calculate accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm}")

conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
print("SVM Confusion Matrix:")
print(conf_matrix_svm)

class_report_svm = classification_report(y_test, y_pred_svm)
print("SVM Classification Report:")
print(class_report_svm)

ConfusionMatrixDisplay.from_estimator(loaded_model, X_test_scaled, y_test)
plt.show()

# Convert y_test and y_pred to 1s and 0s (True -> 1, False -> 0)
y_test_binary = y_test.astype(int)
y_pred_binary = pd.Series(y_pred_svm).astype(int)

# Replace non-finite values with 0 (or another appropriate value)
y_test_binary = y_test_binary.replace([np.inf, -np.inf, np.nan], 0)
y_pred_binary = y_pred_binary.replace([np.inf, -np.inf, np.nan], 0)

# Create predictions dataframe
predictions_df = pd.DataFrame({
    'Actual': y_test_binary,
    'Predicted': y_pred_binary
})

# Ensure values are saved as integers
predictions_df = predictions_df.astype(int)

# Save predictions to CSV
predictions_df.to_csv('predictions.csv', index=False)
print("Predictions telah disimpan ke 'predictions.csv'")

# Download the file
files.download('predictions.csv')