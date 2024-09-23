import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import tkinter as tk
from tkinter import messagebox

# Function to take user input for a new sample from Tkinter fields
def get_user_input():
    try:
        age = int(entry_age.get())
        sex = int(entry_sex.get())
        diet = int(entry_diet.get())
        family_history = int(entry_family_history.get())
        hydration_level = float(entry_hydration_level.get())
        bmi = float(entry_bmi.get())
        exercise_frequency = int(entry_exercise_frequency.get())
        smoking_status = int(entry_smoking_status.get())
        previous_kidney_stones = int(entry_previous_kidney_stones.get())
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter valid data in all fields.")
        return None
    
    user_data = {
        'age': [age],
        'sex': [sex],
        'diet': [diet],
        'family_history': [family_history],
        'hydration_level': [hydration_level],
        'bmi': [bmi],
        'exercise_frequency': [exercise_frequency],
        'smoking_status': [smoking_status],
        'previous_kidney_stones': [previous_kidney_stones]
    }

    return pd.DataFrame(user_data)

# Function to predict kidney stones based on user input and display result
def predict_kidney_stones():
    new_data = get_user_input()
    if new_data is not None:
        new_prediction = best_model.predict(new_data)
        if new_prediction[0] == 1:
            result.set("The model predicts that you are likely to develop kidney stones.")
        else:
            result.set("The model predicts that you are unlikely to develop kidney stones.")
        update_plot(new_data, new_prediction)

# Function to update the scatter plot with user input
def update_plot(new_data, new_prediction):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['age'], df['kidney_stones'], color='blue', label='Actual', alpha=0.5)
    plt.scatter(X_test['age'], y_pred, color='red', label='Predicted', alpha=0.7)
    plt.scatter(new_data['age'], new_prediction, color='green', s=100, label='User Input', edgecolor='black', zorder=5)
    plt.title('Scatter Plot of Actual vs. Predicted Kidney Stones with User Input')
    plt.xlabel('Age')
    plt.ylabel('Kidney Stones (1: Yes, 0: No)')
    plt.legend()
    plt.grid()
    plt.show()

# Sample data
data = {
    'age': [25, 30, 45, 60, 29, 54, 47, 38, 23, 31, 35, 50, 28, 52, 40, 61],
    'sex': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    'diet': [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
    'family_history': [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    'hydration_level': [2.0, 1.5, 1.0, 0.5, 2.5, 2.0, 1.5, 1.0, 2.5, 2.0, 3.0, 1.0, 1.5, 2.5, 1.0, 1.5],
    'bmi': [22.0, 25.0, 30.0, 28.0, 21.0, 27.0, 31.0, 24.0, 22.0, 26.0, 29.0, 23.0, 25.0, 30.0, 28.0, 24.0],
    'exercise_frequency': [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
    'smoking_status': [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
    'previous_kidney_stones': [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0],
    'kidney_stones': [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df[['age', 'sex', 'diet', 'family_history', 'hydration_level', 'bmi', 'exercise_frequency', 'smoking_status', 'previous_kidney_stones']]
y = df['kidney_stones']

# Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Create and fit the Random Forest Classifier with hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Tkinter GUI Setup
root = tk.Tk()
root.title("Kidney Stones Prediction")

# Creating input fields
fields = ['Age', 'Sex (1: Male, 0: Female)', 'Diet (1: High salt, 0: Low salt)', 'Family History (1: Yes, 0: No)',
          'Hydration Level (liters)', 'BMI', 'Exercise Frequency (1: Regular, 0: Rarely)', 'Smoking Status (1: Smoker, 0: Non-smoker)',
          'Previous Kidney Stones (1: Yes, 0: No)']

entries = []
for i, field in enumerate(fields):
    label = tk.Label(root, text=field)
    label.grid(row=i, column=0, padx=10, pady=5)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)

entry_age, entry_sex, entry_diet, entry_family_history, entry_hydration_level, entry_bmi, entry_exercise_frequency, entry_smoking_status, entry_previous_kidney_stones = entries

# Result Label
result = tk.StringVar()
result_label = tk.Label(root, textvariable=result, font=("Helvetica", 12))
result_label.grid(row=10, column=0, columnspan=2, padx=10, pady=20)

# Predict Button
predict_button = tk.Button(root, text="Predict", command=predict_kidney_stones)
predict_button.grid(row=9, column=0, columnspan=2, padx=10, pady=10)

# Start Tkinter event loop
root.mainloop()
