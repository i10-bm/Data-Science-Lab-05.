import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Core Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =====================================================================
# Step 1: Data Acquisition & Synthetic Titanic Generation
# =====================================================================
print("==================================================")
print("STEP 1: Data Loading & Exploration")
print("==================================================")

# Setting a random seed for reproducible synthetic Titanic dataset matching structure
np.random.seed(42)
num_passengers = 891

data = {
    'PassengerId': range(1, num_passengers + 1),
    'Survived': np.random.choice([0, 1], size=num_passengers, p=[0.62, 0.38]),
    'Pclass': np.random.choice([1, 2, 3], size=num_passengers, p=[0.24, 0.21, 0.55]),
    'Sex': np.random.choice(['male', 'female'], size=num_passengers, p=[0.65, 0.35]),
    'Age': np.random.choice([np.nan, 22, 38, 26, 35, 54, 2, 27, 14, 4], size=num_passengers),
    'SibSp': np.random.choice([0, 1, 2, 3], size=num_passengers, p=[0.68, 0.23, 0.05, 0.04]),
    'Parch': np.random.choice([0, 1, 2], size=num_passengers, p=[0.76, 0.13, 0.11]),
    'Fare': np.random.exponential(scale=32.0, size=num_passengers),
    'Embarked': np.random.choice(['S', 'C', 'Q', np.nan], size=num_passengers, p=[0.72, 0.18, 0.09, 0.01])
}

df = pd.DataFrame(data)
print("--- Sample Raw Titanic Dataset Snapshot ---")
print(df.head())
print(f"\nDataset Shape: {df.shape}")


# =====================================================================
# Step 2: Data Preprocessing & Cleaning (Handling Missing Values)
# =====================================================================
print("\n==================================================")
print("STEP 2: Data Preprocessing & Feature Engineering")
print("==================================================")

# 1. Fill missing numerical Age values with the median column value
df['Age'] = df['Age'].fillna(df['Age'].median())

# 2. Fill missing categorical Embarked variables with the most frequent value (mode)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 3. Categorical Encoding: Transform string columns to numerical features
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# 4. Feature Selection: Drop arbitrary primary keys that contain no predictive context
X = df.drop(columns=['PassengerId', 'Survived'])
y = df['Survived']

print("Processed input features layout:")
print(X.head())


# =====================================================================
# Step 3: Train-Test Splitting and Scaling
# =====================================================================
print("\n==================================================")
print("STEP 3: Train-Test Split and Standardization")
print("==================================================")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training matrices dimensions: {X_train_scaled.shape}")
print(f"Testing matrices dimensions: {X_test_scaled.shape}")


# =====================================================================
# Step 4: Model Training and Multi-Algorithm Evaluation
# =====================================================================
print("\n==================================================")
print("STEP 4: Model Training and Multi-Algorithm Comparison")
print("==================================================")

models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n>>> Running Core Model Pipeline: {name} <<<")
    # Tree models do not strictly require scaled features, but passing scaled for evaluation equality
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    
    print(f"Accuracy Performance Score: {acc:.4f}")
    print("Confusion Matrix Layout:")
    print(confusion_matrix(y_test, preds))
    print("Detailed Classification Breakdown Report:")
    print(classification_report(y_test, preds))


# =====================================================================
# Step 5: Hyperparameter Tuning via Grid Search (Random Forest)
# =====================================================================
print("\n==================================================")
print("STEP 5: Hyperparameter Tuning via Grid Search Optimization")
print("==================================================")

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5, 10]
}

rf = RandomForestClassifier(random_state=42)
gs = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
gs.fit(X_train_scaled, y_train)

print(f"Optimized Parameters Selected: {gs.best_params_}")
best_model = gs.best_estimator_
best_preds = best_model.predict(X_test_scaled)
print(f"Optimized Random Forest Test Accuracy: {accuracy_score(y_test, best_preds):.4f}")


# =====================================================================
# Step 6: Model Persistence & Visual Report Saving
# =====================================================================
print("\n==================================================")
print("STEP 6: Saving Best Model & Generating Summary Chart")
print("==================================================")

# Save the absolute best hyperparameter configured estimator model to disk
output_model_name = 'titanic_best_model.joblib'
joblib.dump(best_model, output_model_name)
print(f"[SUCCESS] Model saved successfully as '{output_model_name}'")

# Generate a visual summary plot comparing algorithm accuracies for your report
plt.figure(figsize=(8, 4))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette='viridis')
plt.ylim(0, 1.0)
plt.ylabel('Test Accuracy Score')
plt.title('Titanic Algorithm Comparison Matrix')
for index, value in enumerate(results.values()):
    plt.text(index, value + 0.02, f"{value:.4f}", ha='center')
plt.tight_layout()

# Save image file directly to workspace directory
plot_filename = "titanic_model_comparison.png"
plt.savefig(plot_filename)
plt.close()
print(f"[ALL DONE] Performance bar chart saved directly as '{plot_filename}'")
print("==================================================")