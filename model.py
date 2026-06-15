import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# Download dataset
path = kagglehub.dataset_download("yasserh/titanic-dataset")

# Find the CSV file
files = os.listdir(path)
print("Files in dataset:", files)

# Load dataset (adjust filename if needed)
data = pd.read_csv(os.path.join(path, "Titanic-Dataset.csv"))

# Select useful columns
data = data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

# Handle missing values
data['Age'].fillna(data['Age'].mean(), inplace=True)

# Encode categorical data
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])

# Split data
X = data.drop('Survived', axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)