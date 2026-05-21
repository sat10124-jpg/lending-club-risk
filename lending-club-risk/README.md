##  Executive Summary & Performance
* **The Goal:** Build a production-ready predictive model to flag rare-event borrower defaults (`early_default`).
* **The Challenge:** Manage extreme class imbalance (**only 5.3% of borrowers default**) and detect invisible structural flaws within the raw data.
* **The Result:** Engineered an optimized Logistic Regression classifier achieving a **0.71 ROC AUC score** after successfully neutralizing severe target leakage that artificially inflated baseline evaluation metrics.

---

## Core Data Insights (EDA)
Before training the predictive model, Exploratory Data Analysis (EDA) was performed via `eda_analysis.py` to isolate specific macroeconomic risk drivers:
* **FICO Score Impact:** Real-world credit risk mechanics were visually confirmed. The lowest FICO score bin (660–680) demonstrated the highest historical default concentration, while the highest bin (820–840) registered the lowest default rates.
* **Loan Purpose Exposure:** Specific high-risk borrowing categories emerged. Loans structured for **small businesses, moving expenses, and renewable energy ventures** generated the highest baseline default frequencies across the entire platform.

---

## 🛠️ Data Engineering & Mitigation Pipeline

### 1. Preprocessing & Imputation (`data_preprocessing.py`)
* Processed **933,160 rows** and 37 dimensions.
* **Feature Engineering:** Collapsed highly collinear structural metrics (`fico_range_low` & `fico_range_high`) into a centralized `fico_average` variable. Retained fine-grained categorical detail by prioritizing `sub_grade` over high-level `grade`.
* **Missing Value Recovery:** Formulated an imputation strategy for `mths_since_last_delinq` (49.7% missing). Instead of dropping the data, missing values were treated as a sign of perfect credit health; engineered a binary `has_delinq_history` tracker and filled blanks with a distinct `-1` placeholder.

### 2. The Target Leakage Trap (`train_classification.py`)
* **The Discovery:** A baseline unscaled model registered an impossibly perfect **0.96 ROC AUC**. Inspection of feature coefficients exposed severe data leakage: the historical feature `loan_status_Fully Paid` carried an overwhelming weight of `-5.12`. 
* **The Fix:** Completely purged the `loan_status` variable from the training matrix. Leaving it in would mean giving the model the answers to the final exam using post-facto data that is impossible to know when a live borrower applies for a loan. 

### 3. Feature Scaling & Class Balancing
* Leveraged **Scikit-Learn's `StandardScaler`** to stabilize numerical variances between high-magnitude features (e.g., `annual_inc`) and lower bounds (e.g., `fico_average`).
* Enforced **`class_weight='balanced'`** to heavily penalize the model for missing rare-event defaults, driving Class 1 recall to **68%**.

---

## Final Model Evaluation

```text
=== OPTIMIZED CLASSIFICATION REPORT ===
              precision    recall  f1-score   support

           0       0.97      0.62      0.76    176762
           1       0.09      0.68      0.16      9870

    accuracy                           0.63    186632
   macro avg       0.53      0.65      0.46    186632
weighted avg       0.93      0.63      0.73    186632

ROC AUC Score: 0.7077