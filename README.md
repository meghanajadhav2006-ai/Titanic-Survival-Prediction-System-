# 🚢 Titanic Survival Prediction System

## 📌 Overview

The **Titanic Survival Prediction System** is a machine learning project that predicts whether a passenger survived the Titanic disaster using passenger information such as age, gender, passenger class, and fare.

The project demonstrates a machine learning workflow involving data preprocessing, exploratory data analysis (EDA), model training, and evaluation using Logistic Regression.

## 🎯 Objectives

- Develop a machine learning model for binary classification.
- Analyze passenger data to identify survival patterns.
- Handle missing values and encode categorical variables.
- Train and evaluate a Logistic Regression model.
- Visualize relationships between passenger features and survival rates.

## 🚀 Features

- Data preprocessing and cleaning
- Missing-value handling
- Categorical feature encoding
- Exploratory data analysis using graphs
- Logistic Regression classification
- Model evaluation using accuracy
- Visualization of passenger age, fare, gender, and passenger class

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## 📂 Project Structure

```text
titanic-survival-prediction/
├── README.md
├── model.py
├── Titanic-Dataset.csv
├── output.png
└── images/
    ├── fare-distribution.png
    ├── age-distribution.png
    ├── survival-by-class.png
    ├── survival-by-gender.png
    ├── roc-curve.png
    └── regression-plot.png
```

## 📊 Dataset

**Dataset:** Titanic Passenger Dataset from Kaggle

The dataset contains passenger information, including age, gender, passenger class, fare, and survival status.

These features are used to explore survival patterns and train the classification model.

## 📈 Exploratory Data Analysis and Visualizations

### 1. Fare Distribution

<img width="823" height="521" alt="FA1-DS (3)" src="https://github.com/user-attachments/assets/84f437f6-6e06-4009-b5ab-dd2fbdf1d80c" />

The histogram illustrates the distribution of passenger fares. Most passengers paid relatively lower fares, while a smaller number paid substantially higher fares.

### 2. Age Distribution

<img width="797" height="525" alt="FA1-DS (2)" src="https://github.com/user-attachments/assets/8a93edad-db74-4257-bdfa-788ad1d5636d" />

The histogram shows the age distribution of passengers. It helps identify the age groups represented in the dataset and understand the distribution of passenger ages.

### 3. Survival Rate by Passenger Class

<img width="692" height="467" alt="FA1-DS (1)" src="https://github.com/user-attachments/assets/9d82f727-b658-43af-a9fb-2ca331935fe8" />

The visualization compares survival rates across first, second, and third class passengers. In the displayed results, first-class passengers have the highest survival rate, while third-class passengers have the lowest.

### 4. Survival Rate by Gender

<img width="713" height="467" alt="FA1-DS" src="https://github.com/user-attachments/assets/b2467c18-2246-4236-be70-61850a99b5e1" />


The chart compares survival rates between male and female passengers. The displayed results indicate a substantially higher survival rate among female passengers.

### 5. ROC Curve

<img width="550" height="528" alt="FA1-DS (5)" src="https://github.com/user-attachments/assets/68d0b80b-eb1d-498f-b519-df0c7403f144" />



### 6. Regression Visualization

<img width="639" height="408" alt="FA1-DS (4)" src="https://github.com/user-attachments/assets/44978916-fe5c-49ca-a999-84d77d2bf4b9" />


## ⚙️ Model and Evaluation

**Algorithm:** Logistic Regression

**Task:** Binary classification

The model uses selected passenger features to predict survival status.

**Model Accuracy:** Approximately 80% (reported project result; verify against your final test run).

Accuracy alone does not fully describe classification performance. Additional metrics such as precision, recall, F1-score, and ROC-AUC can be included if calculated.

## ▶️ How to Run

1. Install Python.
2. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```

3. Ensure `model.py` and `Titanic-Dataset.csv` are in the project directory.
4. Run the model:

   ```bash
   python model.py
   ```

5. Review the generated prediction output and visualizations.

## 📚 Learning Outcomes

- Understanding the machine learning workflow
- Performing data cleaning and preprocessing
- Applying categorical encoding
- Training a Logistic Regression classifier
- Conducting exploratory data analysis
- Interpreting classification results and visualizations

## 🔮 Future Improvements

- Compare Logistic Regression with Random Forest and other classifiers.
- Add confusion matrix, precision, recall, and F1-score.
- Generate an ROC curve from actual model predictions.
- Improve feature engineering and missing-value handling.
- Develop an interactive prediction interface using Streamlit or Flask.

## 👩‍💻 Author

**Meghana Jadhav**

Electronics and Telecommunication Engineering  
Pimpri Chinchwad College of Engineering (PCCOE), Pune
