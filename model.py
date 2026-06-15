import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('train.csv')

# Select useful columns
data = data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

# Handle missing values
data['Age'].fillna(data['Age'].mean(), inplace=True)

# Convert categorical data
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