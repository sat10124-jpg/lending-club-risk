import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

print("🚀 Loading cleaned dataset...")
df = pd.read_csv('lc_loan_clean.csv')

# 1. Separate features and target
X = df.drop(columns=['early_default', 'return', 'loan_status'])
y = df['early_default']

# 2. One-Hot Encode categorical variables
print("🔀 Encoding categorical variables...")
X = pd.get_dummies(X, drop_first=True)

# 3. Train/Test Split
print("⚖️ Splitting data into Train and Test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. FEATURE SCALING: Essential for stable Logistic Regression coefficients!
print("📏 Scaling numerical features...")
scaler = StandardScaler()
# Scale the training features, then apply the exact same mapping to test features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train the balanced model on scaled data
print("🤖 Training balanced Logistic Regression model on scaled data...")
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Evaluate
print("\n=== OPTIMIZED CLASSIFICATION REPORT ===")
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]):.4f}")

# 7. EXPOSE THE RISK DRIVERS (Find the professor's hidden feature)
print("\n🔍 TOP 10 STRONGEST PREDICTORS OF DEFAULT RISK:")
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})
# Sort by absolute value to find the strongest positive and negative drivers
coefficients['Abs_Coef'] = coefficients['Coefficient'].abs()
print(coefficients.sort_values(by='Abs_Coef', ascending=False).head(10))