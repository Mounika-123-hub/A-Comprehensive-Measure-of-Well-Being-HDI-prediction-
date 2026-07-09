import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import joblib

# Read the dataset
Development = pd.read_csv("dataset/HDI.csv")
# Remove duplicate rows
Development = Development.drop_duplicates()

print("Dataset shape after removing duplicates:")
print(Development.shape)


for col in Development.columns:
    if "Life Expectancy" in col:
        print(col)

for col in Development.columns:
    if "Expected Years" in col:
        print(col)

for col in Development.columns:
    if "Mean Years" in col:
        print(col)

for col in Development.columns:
    if "Gross National Income" in col:
        print(col)

for col in Development.columns:
    if "Human Development Index" in col:
        print(col)

# Display first five rows
print("First 5 Rows:")
print(Development.head())

# Display dataset shape
print("\nDataset Shape:")
print(Development.shape)

# Display dataset information
print("\nDataset Information:")
Development.info()

# Check missing values
print("\nMissing Values:")
print(Development.isnull().sum())

# Display first 50 column names
print("\nFirst 50 Column Names:")
for i, column in enumerate(Development.columns[:50], start=1):
    print(f"{i}. {column}")

# Display all columns containing 2021
print("\nColumns containing 2021:")
for col in Development.columns:
    if "2021" in col:
        print(col)
        # -------------------------------
# Selecting Dependent and Independent Variables
# -------------------------------

# Select only the required columns
Development = Development[
    [
        "Country",
        "Life Expectancy at Birth (2021)",
        "Expected Years of Schooling (2021)",
        "Mean Years of Schooling (2021)",
        "Gross National Income Per Capita (2021)",
        "Human Development Index (2021)"
    ]
]
Development.columns = [
    "Country",
    "Life Expectancy at Birth",
    "Expected Years of Schooling",
    "Mean Years of Schooling",
    "Gross National Income Per Capita",
    "Human Development Index"
]

print("\nSelected Dataset:")
print(Development.head())

# -------------------------------
# Checking and Handling Null Values
# -------------------------------

print("\nMissing Values in Selected Dataset:")
print(Development.isnull().sum())

# Remove missing values
Development = Development.dropna()

print("\nDataset Shape After Removing Null Values:")
print(Development.shape)

# -------------------------------
# Independent Variables (X)
# -------------------------------
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

Development["Country"] = label_encoder.fit_transform(Development["Country"])

print("\nAfter Label Encoding:")
print(Development.head())

X = Development[
    [
        "Life Expectancy at Birth",
        "Expected Years of Schooling",
        "Mean Years of Schooling",
        "Gross National Income Per Capita"
    ]
]
print("\nNull Values in X:")
print(X.isnull().sum())

y = Development["Human Development Index"]

print("\nIndependent Variables (X):")
print(X.head())

print("\nDependent Variable (y):")
print(y.head())
# -------------------------------
# Split the dataset into Training and Testing sets
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

print("\nTraining Target Shape:")
print(y_train.shape)

print("\nTesting Target Shape:")
print(y_test.shape)
# ---------------------------------------
# Fit the Linear Regression Model
# ---------------------------------------

# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\nLinear Regression Model Trained Successfully!")

# Display model coefficients
print("\nModel Coefficients:")
print(model.coef_)

# Display model intercept
print("\nModel Intercept:")
print(model.intercept_)
# ---------------------------------------
# Predicting the Results
# ---------------------------------------

# Predict HDI values for the test data
y_pred = model.predict(X_test)
# Model Evaluation

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("----------------------------")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)
comparison = pd.DataFrame({
    "Actual HDI": y_test.values,
    "Predicted HDI": y_pred
})

print("\nActual vs Predicted")
print(comparison.head(10))

print("\nFirst 10 Predicted HDI Values:")
print(y_pred[:10])

print("\nFirst 10 Actual HDI Values:")
print(y_test.head(10).values)
# Save the trained model
joblib.dump(model, "model/hdi_model.pkl")

print("\nModel saved successfully as model/hdi_model.pkl")
# -------------------------------
# DATA VISUALIZATION
# -------------------------------

# Select first 20 rows
data1 = Development.head(20)

# Display unique country names
print("\nUnique Country Names:")
print(data1["Country"].unique())

# Create a new dataframe with required columns
visual_data = data1[[
    "Country",
    "Mean Years of Schooling",
    "Life Expectancy at Birth",
    "Human Development Index"
]]

# Remove missing values
visual_data = visual_data.dropna()

# --------------------------------
# 1. Mean Years of Schooling vs HDI
# --------------------------------
plt.figure(figsize=(8,5))
sns.stripplot(
    x="Mean Years of Schooling",
    y="Human Development Index",
    data=visual_data
)
plt.title("Mean Years of Schooling vs HDI")
plt.tight_layout()
plt.show()

# --------------------------------
# 2. Life Expectancy vs HDI
# --------------------------------
plt.figure(figsize=(8,5))
sns.stripplot(
    x="Life Expectancy at Birth",
    y="Human Development Index",
    data=visual_data
)
plt.title("Life Expectancy vs HDI")
plt.tight_layout()
plt.show()

# --------------------------------
# 3. Correlation Heatmap
# --------------------------------
corr = visual_data.drop(columns=["Country"]).corr()

plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save the trained model
with open("model/hdi_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully as model/hdi_model.pkl")