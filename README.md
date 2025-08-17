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
  Train vs Test gap is small, which means the model generalizes consistently. Both ROC and PR scores are low, indicating the features don’t carry much signal.  Features may be uninformative or noisy or maybe there are nonlinear relationships that logistic regression can't identify

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
  - PR-AUC: **0.127**   Did not outperform Logistic Regression → suggests limited signal in data.  

---

## Challenges

- **Main Challenge**:  
  - The dataset is **synthetic**, and multivariate correlations are weak.  
  - As a result, Logistic Regression cannot establish strong decision boundaries, and even more complex models (e.g., XGBoost, Random Forest) plateau at a similar performance level.  

- **Generalization**:  
  - The **train–test gap** is small, indicating the models generalize consistently.  
  - However, both **ROC** and **PR** scores remain low, showing that the features carry little predictive signal.  

- **Possible Causes**:  
  - Features may be **uninformative or noisy**.  
  - The **target label** may be close to random, which is common in synthetic datasets.  
  - Even when testing models that can capture nonlinear relationships (XGBoost, Random Forest), the results remained nearly identical—further supporting the idea that the dataset lacks meaningful structure for prediction.  

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

To improve business operations, we can take the following steps:  

1. **Monitor High-Risk Products**  
   - Products with unusually high return rates should be flagged for review.  
   - Investigate potential **quality issues**, **misleading product descriptions**, or **mismatched customer expectations**.  

2. **Focus on Price-Sensitive Segments**  
   - Returns are higher among **expensive items** and **first-time buyers**.  
   - Strategies: provide clearer sizing guides, detailed specifications, or stricter pre-purchase confirmations for costly products.  

3. **Introduce Customer Segmentation**  
   - Segment customers by purchase history (first-time vs. repeat), spending level, and product category preferences.  
   - Tailor marketing, promotions, and post-purchase communications to each segment to reduce return likelihood.  

4. **Category-Specific Models**  
   - Since return behavior differs across categories, consider building **separate predictive models** for each product category.  
   - For example, clothing may require size-related features, while electronics may depend more on price and warranty details.  

5. **Risk Tiering**  
   - Develop a **tiered risk scoring system** (e.g., low, medium, high) for orders based on predicted return probability.  
   - Use tiers to drive **business rules** such as:  
     - High-risk orders → additional verification, targeted product guidance.  
     - Medium-risk orders → optional support nudges (e.g., post-purchase tips).  
     - Low-risk orders → standard workflow, less need for intervention.  
   - This ensures resources are allocated efficiently while minimizing unnecessary interventions.  

## ⚙️ How to Run the Code  

The project is organized around a single Jupyter Notebook for reproducibility.  

```bash
# Clone the repository
git clone https://github.com/your-repo/product-return-prediction.git
cd product-return-prediction

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter and open the main notebook
jupyter notebook python/notebooks/main.ipynb
```

Inside `main.ipynb`, you will find:  
- **EDA section** → dataset exploration, distributions, and target analysis.  
- **Feature engineering** → log transforms, interaction terms, and target encodings.  
- **Modeling** → training Logistic Regression, Random Forest, and XGBoost with imbalance handling.  
- **Evaluation** → metrics (F1, ROC-AUC, PR-AUC, top-k lift).  
- **Interpretation** → feature importance and key insights.  
