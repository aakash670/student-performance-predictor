import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = pd.read_csv('students_data.csv')

# Features and target
X = data[['hours_studied', 'attendance', 'assignments_submitted']]
y = data['final_score']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict
predictions = model.predict(x_test)

# Bar Graph: Actual vs Predicted
indices = np.arange(len(y_test))  # X axis positions
bar_width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(indices, y_test.values, width=bar_width, color='blue', label='Actual')
plt.bar(indices + bar_width, predictions, width=bar_width, color='red', label='Predicted')

plt.xlabel('Student Index')
plt.ylabel('Final Score')
plt.title('Actual vs Predicted Final Scores')
plt.xticks(indices + bar_width / 2, [f'Student {i+1}' for i in indices])
plt.legend()
plt.tight_layout()
plt.show()
