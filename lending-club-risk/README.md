# Lending Club Credit Risk Project

This is a project I built for America on Tech (Tech 360) to look at credit risk using historical peer-to-peer loan data from Lending Club. The goal was to clean a massive dataset of nearly 1 million rows, figure out what actually makes a borrower likely to default, and train a model to catch those risky loans early.

## Summary of my project
* **The Goal:** Build a machine learning model to predict if a borrower will default (`early_default`).
* **The problem with the dataset** Only about 5.3% of the people in this dataset actually default. Because defaults are so rare, a lazy model could just guess "safe" every time, get 95% accuracy, and completely fail at its job.
* **My First attempt:** My first attempt at the model gave me an impossibly high score (a 0.96 ROC AUC). When I checked why, I realized the dataset had a massive case of data leakage. It included a column called `loan_status_Fully Paid`. In the real world, you obviously don't know if a loan is fully paid back when someone first applies for it! Leaving that in was like giving the model an answer key. 
* **The Fix:** I completely stripped out the `loan_status` variables, forcing the model to rely only on background info like income and credit history. The real, un-cheated model ended up with a solid **0.71 ROC AUC score**.



## My process

### 1. Cleaning & Preprocessing in `data_preprocessing.py`
* Processed all **933,160 rows** of data.
* **Feature Engineering:** The dataset had separate columns for the low and high end of a borrower's FICO score. Since they track the same thing, I averaged them into one single `fico_average` column and dropped the duplicates. I also kept the specific `sub_grade` column over the broad `grade` column to keep finer details.
* **Handling Missing Values:** About half of the dataset was missing values for `mths_since_last_delinq`. Instead of throwing that data away, I realized that missing data here actually means the person has never had a delinquency . I made a new true/false column called `has_delinq_history` and filled the blanks with `-1`.

### 2. Training & Scaling in `train_classification.py`
* Because variables like income ($100k+) and FICO scores (~700) use  different scales, I used Scikit-Learn's `StandardScaler` to normalize the data so the model wouldn't get confused by the large numbers.
* To fix the 5.3% default imbalance, I used class_weight='balanced'. This essentially told the model: "Missing a default is a massive deal, so pay extra attention to them." This bumped the model's recall up to 68% (meaning it successfully catches 68% of all actual defaults), while maintaining an overall 0.71 ROC AUC score.



## How do we know how to predict a default? 

After training the Logistic Regression model, I pulled the strongest coefficients to see what variables impacted the risk the most:

1. **`installment` (+1.06):** This was the single biggest predictor of default. The higher the monthly payment amount, the more stressed the borrower's budget is, making them much more likely to default.
2. **`funded_amnt` / `loan_amnt` (-0.55 / -0.45):**  if the monthly payment stays the same, a larger overall loan amount slightly lowered default risk (likely because it means the loan is spread out over a longer, more manageable timeline).
3. **Sub-Grades (B5 to D1):** The model automatically picked up on the fact that lower bank tiers sharply increased default risk, perfectly matching how real financial institutions price out risk.



##  How to Run It
1. Download the raw dataset (`lc_loan.csv`) from this google drive link: https://drive.google.com/file/d/18wB8zJbPjKOBS41XjUInHmZkDUgtYMKy/view?usp=sharing
2. Move the downloaded `lc_loan.csv` file directly into this project's root folder.
3. Run the cleaning script:
   ```bash
   python data_preprocessing.py
