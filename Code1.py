# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Data Acquisition

# Load your dataset (replace 'himanshu_dataset.csv' with your data file)
data = pd.read_csv('himanshu_dataset.csv')

# Step 2: Exploratory Data Analysis (EDA)

# Display basic information about the dataset
print("Dataset Info:")
print(data.info())

# Display summary statistics of numerical columns
print("\nSummary Statistics:")
print(data.describe())

# Visualize the distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='placement', data=data)
plt.title('Distribution of Academic Placement')
plt.xlabel('Placement')
plt.ylabel('Count')
plt.show()

# Step 3: Data Preprocessing

# Handle missing values (if any)
data.dropna(inplace=True)

# Encode categorical features (if any) using one-hot encoding or label encoding

# Define your features (X) and target (y)
X = data.drop('placement', axis=1)  # Replace 'placement' with your target column name
y = data['placement']

# Step 4: Data Splitting

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training

# Initialize and train a machine learning model (Random Forest classifier)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Model Evaluation

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Generate a classification report for more detailed evaluation
report = classification_report(y_test, y_pred)
print('Classification Report:\n', report)

# Step 7: Predict on New Data (Optional)

# To use the trained model for predictions on new data, create a new DataFrame with the features and use model.predict(new_data)
# For example:
# new_data = pd.DataFrame({'feature1': [value1], 'feature2': [value2], ...})
# prediction = model.predict(new_data)
