#  Product Return Prediction — README Report

##  Approach

My approach to this project was to treat it as if I were working with a real-world e-commerce dataset.  

1. I started with a **clean EDA** to understand the structure of the data.  
2. I then engineered features that seemed meaningful for predicting returns:
   - Log price
   - Weekend flag
   - Product frequency
   - Category z-scores
   - Interaction terms
   - Target-encoded return rates  
3. I benchmarked two main types of models:
   - **Logistic Regression** as a fast, interpretable baseline.
   - **XGBoost** as a flexible tree-based learner with imbalance handling and early stopping.  
   - I also tried **Random Forest** for comparison.  

I carefully split train/validation/test, avoided leakage in target encoding with out-of-fold estimates, and chose operating thresholds based on **validation F1** (not default probability cutoffs). I also looked at **capacity metrics (top-k lift)**, since in practice a business might only investigate the riskiest 5–15% of orders.

---

##  Exploratory Data Analysis (EDA)

### Dataset Overview
- **Rows**: 60,000 orders  
- **Columns**: 7  
- **Target variable**: `returned` (binary, 1 = returned, 0 = not returned)  
- **Return rate**: ~8.93% (≈ 1 in 11 orders returned)  

| Column                  | Type   | Description                           | Notes                                             |
|--------------------------|--------|---------------------------------------|--------------------------------------------------|
| order_id                | object | Unique order identifier               | 60,000 unique values                              |
| product_id              | object | Unique product identifier             | 200 products, ~300 orders per product             |
| product_category        | object | Product category                      | 7 categories                                      |
| price                   | float  | Product price in USD                  | \$6.47 – \$1415.64, right-skewed                  |
| is_first_time_customer  | int    | 1 = first-time buyer                  | 30% first-time customers                          |
| order_day_of_week       | int    | Day of order (0=Mon, 6=Sun)           | Balanced across weekdays                          |
| returned                | int    | Target (1 = returned, 0 = not returned) | 9% positive class → imbalanced                    |

 No missing values  
 No duplicate rows  
 Clean feature types  

---

### Target Distribution
- Only **~9% of orders are returned** → **imbalanced classification problem**.  
- Accuracy alone is misleading; metrics like **precision, recall, F1, ROC-AUC, PR-AUC** are more informative.  

---

### Numerical Feature: Price
- **Range**: \$6.47 – \$1415.64  
- **Mean**: \$50.51  
- **Median**: \$38.32  
- **Distribution**: Right-skewed (most products are < \$100).  
- **Returns**: Expensive products are slightly more likely to be returned.  
 Applied **log-transform** (`log_price`) to normalize.  

---

### Boolean Feature: First-Time Customer
- 30% are first-time buyers.  
- Return rates:  
  - First-time buyers → ~11%  
  - Repeat buyers → ~8%  

 First-time customers are more likely to return products.  

---

### Temporal Feature: Order Day of Week
- Days 0–6 (Monday–Sunday).  
- Return rates:  
  - Mon–Thu → ~8–9%  
  - Sat–Sun → ~10–10.5%  

 Weekend purchases have higher return rates, possibly indicating impulsive buys. 

---

### Categorical Feature: Product Category
- 7 categories.  
- Used for **price normalization** within category.  

---

### Categorical Feature: Product ID
- 200 unique products, ~300 orders each.  
- High cardinality → unsuitable for one-hot encoding.  
- Used **target encoding** instead.  

---

### Top 20 Products by Return Rate
| product_id | orders | returns | return_rate |
|------------|--------|---------|-------------|
| P-203      | 300    | 40      | 13.3%       |
| P-230      | 300    | 40      | 13.3%       |
| P-199      | 300    | 39      | 13.0%       |
| P-245      | 300    | 38      | 12.7%       |
| P-288      | 300    | 38      | 12.7%       |
| ...        | ...    | ...     | ...         |

 Certain products consistently have above-average return rates (>13%).  

---

##  Feature Engineering

### Engineered Features
- `log_price` → handles skew in raw price.  
- `is_weekend` → captures weekend buying behavior.  
- `product_freq`, `log_product_freq` → product popularity.  
- `price_bucket` → quantile bins (Low / Med / High / Premium).  
- `first_time_x_log_price` → interaction (first-time × price).  
- `price_z_in_cat` → price z-score within category.  
- `product_return_rate` → OOF target encoding for product_id.  
- `category_return_rate` → OOF target encoding for product_category.  

 Together, these features capture **pricing effects, customer behavior, product popularity, and historical return patterns**.

---

##  Modeling & Results

### Logistic Regression (Baseline)
- **Why**: Fast, interpretable, sets a performance floor.  
- **Threshold**: 0.516 (chosen by max F1 on validation).  
- **Test Metrics**:
  - ROC-AUC: **0.605**
  - PR-AUC: **0.132**
  - F1-score: **0.210**  
- **Capacity View**:
  - Top 5% flagged → Precision = 0.178, Lift = 2.0x  
  - Top 10% flagged → Precision = 0.165, Lift = 1.8x  
  - Top 15% flagged → Precision = 0.158, Lift = 1.8x  

---

### XGBoost
- **Why**: More flexible learner, handles non-linearities.  
- **Test Metrics**:
  - ROC-AUC: **0.575**
  - PR-AUC: **0.119**  
- **Observation**: Did not outperform Logistic Regression → suggests limited signal in data.  

---

### Random Forest
- **Why**: Nonlinear, bagged trees for robustness.  
- **Train Metrics**:
  - ROC-AUC: 0.620  
  - PR-AUC: 0.141  
- **Test Metrics**:
  - ROC-AUC: **0.601**
  - PR-AUC: **0.127**  

---

##  Challenges

- **Main Challenge**:  
  - The dataset is **synthetic**, and multivariate correlations are weak.  
  - As a result, Logistic Regression cannot find strong decision boundaries, and even advanced models are capped around the same level.  

The train-test gap is small, indicating that the model generalizes consistently. Both ROC and PR scores are low, indicating the features don’t carry much signal.  Features may be uninformative or noisy

Or the target label may be close to random (especially common in synthetic data) to see if maybe there are nonlinear relationships that logistic regression can't identify, I also used XGBoost and random Forrest, which  both produced close to exact same results that proves the dataset Features may be uninformative or noisy Or the target label may be close to random
---

##  Key Takeaways

- The **pipeline** is involves:
  - Clean feature engineering
  - Proper handling of class imbalance
  - Avoided leakage with OOF encodings
  - Threshold optimization based on F1
  - Interpreted results with PR-AUC and top-k lift  

- The **main limitation is the data**
- With only 7 simplified features and synthetic labels, predictive power is capped.  


