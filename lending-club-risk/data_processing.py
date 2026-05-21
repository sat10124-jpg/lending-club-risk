import pandas as pd
import numpy as np

def preprocess_lending_data(file_path):
    print("🚀 Loading dataset for preprocessing...")
    df = pd.read_csv(file_path)
    
    # 1. Drop columns suggested by guidelines to simplify the model
    cols_to_drop = ['id', 'earliest_cr_line', 'zip_code', 'issue_d']
    df = df.drop(columns=cols_to_drop)
    print(f" Dropped high-cardinality/unnecessary columns: {cols_to_drop}")
    
    # 2. Feature Engineering: FICO Average
    df['fico_average'] = (df['fico_range_low'] + df['fico_range_high']) / 2
    df = df.drop(columns=['fico_range_low', 'fico_range_high'])
    print(" Engineered 'fico_average' and dropped raw ranges.")
    
    # 3. Handle Missing Values: Delinq Months
    # Create an indicator column before filling missing values
    df['has_delinq_history'] = np.where(df['mths_since_last_delinq'].isnull(), 0, 1)
    # Fill missing values with a distinct placeholder (e.g., -1)
    df['mths_since_last_delinq'] = df['mths_since_last_delinq'].fillna(-1)
    print(" Imputed 'mths_since_last_delinq' with placeholder -1 and created binary history indicator.")
    
    # 4. Handle Missing Values: Employment Length
    # Treat missing employment length as its own category 'Unknown'
    df['emp_length'] = df['emp_length'].fillna('Unknown')
    print(" Filled missing 'emp_length' values with 'Unknown'.")
    
    # 5. Drop one of the redundant grade columns (Keeping sub_grade for finer grain detail)
    df = df.drop(columns=['grade'])
    print(" Dropped redundant 'grade' column (retained 'sub_grade').")
    
    return df

if __name__ == "__main__":
    cleaned_df = preprocess_lending_data('lc_loan.csv')
    
    print("\n=== PREPROCESSED DATASET HEALTH CHECK ===")
    print(f"New Shape: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns")
    print(f"Remaining Missing Values: {cleaned_df.isnull().sum().sum()}")
    
    # Save the processed framework to a new CSV for modeling
    print("\n💾 Saving cleaned dataset to 'lc_loan_clean.csv'...")
    cleaned_df.to_csv('lc_loan_clean.csv', index=False)
    print(" Clean dataset successfully saved!")