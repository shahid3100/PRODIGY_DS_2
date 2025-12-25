import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("train.csv")

# Basic info
print("Dataset Info:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

# ---------------- Data Cleaning ----------------
# Fill missing Age with median
data['Age'].fillna(data['Age'].median(), inplace=True)

# Fill missing Embarked with mode
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Drop Cabin column (too many missing values)
data.drop('Cabin', axis=1, inplace=True)

# ---------------- Exploratory Data Analysis ----------------
sns.set(style="whitegrid")

# 1. Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=data, palette='pastel')
plt.title("Survival Count")
plt.show()

# 2. Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=data)
plt.title("Survival by Gender")
plt.show()

# 3. Survival by Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title("Survival by Passenger Class")
plt.show()

# 4. Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(data['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# 5. Fare vs Survival
plt.figure(figsize=(6,4))
sns.boxplot(x='Survived', y='Fare', data=data)
plt.title("Fare vs Survival")
plt.show()
