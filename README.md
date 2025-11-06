# Machine Learning Classifiers Comparison: Bank Marketing Campaign

A comprehensive machine learning analysis comparing the performance of K-Nearest Neighbors, Logistic Regression, Decision Trees, and Support Vector Machines for predicting bank term deposit subscriptions.

## 📊 Project Overview

This project analyzes data from a Portuguese banking institution's direct marketing campaigns (phone calls) to predict whether clients will subscribe to a term deposit. The goal is to identify the best performing classifier to optimize future marketing campaigns.

**Dataset Source**: [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

**Time Period**: May 2008 - November 2010  
**Total Campaigns**: 17 campaigns  
**Dataset Size**: 41,176 records (after removing 12 duplicates)

Notebook: [prompt_III.ipynb](prompt_III.ipynb)

---

##  Business Objective

**Primary Goal**: Develop a predictive model to identify which bank clients are most likely to subscribe to a term deposit product, enabling the bank to:

1. **Increase Campaign Efficiency**: Prioritize high-potential leads
2. **Reduce Marketing Costs**: Minimize wasted contact attempts with unlikely prospects
3. **Improve Customer Experience**: Avoid repeatedly contacting uninterested customers
4. **Maximize ROI**: Optimize return on investment by increasing the current ~11% success rate

---

## Dataset Description

### Target Variable
- **isSubscribed** (y): Has the client subscribed to a term deposit? (yes/no)
  - **Class Distribution**: 88.73% No, 11.27% Yes (⚠️ **Highly Imbalanced**)

### Features Used (Bank Client Information)
1. **age** (numeric): Client's age
2. **job** (categorical): Type of job (12 categories)
3. **marital** (categorical): Marital status (4 categories)
4. **education** (categorical): Education level (8 categories)
5. **default** (categorical): Has credit in default? (yes/no/unknown)
6. **housing** (categorical): Has housing loan? (yes/no/unknown)
7. **loan** (categorical): Has personal loan? (yes/no/unknown)

### Data Quality
-  No missing values (NaN)
- "Unknown" values present in some categorical features (representing missing data)
-  12 duplicate records removed
-  All data types appropriate

---

## 🔧 Data Preprocessing

### Encoding Strategy
1. **Numeric Features (age)**: 
   - Standardized using `StandardScaler` (mean=0, std=1)
   
2. **Categorical Features**: 
   - One-hot encoded using `OneHotEncoder` with `drop='first'` to avoid multicollinearity
   - Expanded from 7 original features to 28 features after encoding

3. **Target Variable**: 
   - Binary encoded: 'yes' → 1, 'no' → 0

### Train/Test Split
- **Training Set**: 32,940 samples (80%)
- **Testing Set**: 8,236 samples (20%)
- **Stratification**: Applied to maintain class balance (88.73% / 11.27% in both sets)

---

## Models Compared

### 1. Baseline Model
**DummyClassifier (Most Frequent Strategy)**
- Always predicts "no" (majority class)
- **Accuracy**: 88.73%
- **Problem**: 0% recall for subscribers - completely fails to identify potential customers!

### 2. Logistic Regression
- **Baseline Accuracy**: 88.73%
- **ROC-AUC**: 0.6487
- **Issue**: Without tuning, predicts mostly "no" class

### 3. K-Nearest Neighbors (KNN)
- **Baseline Accuracy**: 87.57%
- **ROC-AUC**: ~0.70
- **Note**: Computationally expensive with large datasets

### 4. Decision Tree
- **Baseline Accuracy**: 86.12%
- **ROC-AUC**: ~0.75
- **Issue**: Prone to overfitting (30% train-test accuracy gap before tuning)

### 5. Support Vector Machine (SVM)
- **Baseline Accuracy**: 88.73%
- **Training Time**: 191 seconds (slowest)
- **Issue**: Very slow with large datasets

---

## Key Findings

### Challenge: Class Imbalance Problem
The dataset is **highly imbalanced** (88.73% "no", 11.27% "yes"), which causes:
- Models tend to predict the majority class ("no") to maximize accuracy
- Poor identification of actual subscribers (the minority class)
- Misleading accuracy metrics

### Solution: Hyperparameter Tuning with Class Weights
To address the imbalance, we implemented:

1. **Changed Optimization Metric**: 
   - From `accuracy` to `ROC-AUC` (better for imbalanced data)
   - ROC-AUC measures discrimination ability across all thresholds

2. **Added Class Weights**:
   - `class_weight='balanced'` for Logistic Regression, Decision Tree, and SVM
   - Gives more importance to the minority class (subscribers)

3. **Optimized Hyperparameters**:
   - **Logistic Regression**: Regularization strength (C), penalty type
   - **KNN**: Number of neighbors, weighting scheme
   - **Decision Tree**: Max depth, min samples split/leaf
   - **SVM**: C parameter, kernel type

4. **Computational Optimization**:
   - Used subset sampling for KNN (5,000 samples) and SVM (3,000 samples) during tuning
   - Retrained final models on full dataset with best parameters
   - Reduced CV folds to 2 for faster execution

---

## 🏆 Model Performance Comparison

### Baseline Models (Before Tuning)

| Model                   | Train Time (s) | Test Accuracy | ROC-AUC | F1-Score | Recall |
|-------------------------|----------------|---------------|---------|----------|--------|
| Baseline (Most Frequent)| 0.01           | 88.73%        | 0.50    | 0.00     | 0.00   |
| Logistic Regression     | 0.20           | 88.73%        | 0.65    | 0.00     | 0.00   |
| KNN                     | 0.02           | 87.57%        | 0.70    | Low      | Low    |
| Decision Tree           | 1.67           | 86.12%        | 0.75    | Medium   | Medium |
| SVM                     | 191.25         | 88.73%        | 0.50    | 0.00     | 0.00   |

### After Hyperparameter Tuning (Expected Results)

With class weight balancing and ROC-AUC optimization:
- **Improved F1-Scores**: Better balance between precision and recall
- **Higher Recall**: Better identification of potential subscribers
- **Better ROC-AUC**: Improved discrimination between classes
- **Maintained Accuracy**: While also identifying minority class

---

## Recommendations

### 1. **Model Selection**
Based on the analysis, the recommended model should have:
- ✅ **Highest ROC-AUC score** (best discrimination ability)
- ✅ **Balanced precision and recall** (F1-score)
- ✅ **Good recall for "yes" class** (identify subscribers)
- ✅ **Reasonable training time** (practical deployment)

**Expected Best Performer**: Decision Tree or Logistic Regression (tuned with class weights)

**Immediate Actions:**
1. ✅ Deploy the best performing model in a pilot campaign
2. ✅ Set up monitoring for model performance metrics
3. ✅ Create a feedback loop to retrain model with new data

**Future Enhancements:**
1. **Include additional features**:
   - Campaign-related features (contact duration, number of contacts)
   - Previous campaign outcomes
   - Economic indicators (employment rate, consumer confidence)
2.  **Try ensemble methods**:
   - Random Forest, Gradient Boosting, XGBoost
   - Often perform better than single models
3.  **Customer segmentation**:
   - Build separate models for different customer segments
   - Personalized prediction strategies


---

## Key Metrics to Track

### Model Performance Metrics
1. **ROC-AUC**: Overall discrimination ability
2. **Precision**: Of predicted subscribers, how many actually subscribe?
3. **Recall**: Of actual subscribers, how many did we identify?
4. **F1-Score**: Harmonic mean of precision and recall

### Business Metrics
1. **Conversion Rate**: % of contacted customers who subscribe
2. **Cost per Acquisition**: Marketing cost / number of new subscribers
3. **Campaign ROI**: Revenue from subscriptions / marketing costs
4. **Customer Satisfaction**: Feedback on contact frequency

---

## Technical Details

### Requirements
```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

### File Structure
```
ml-classifiers-comparison/
├── README.md                          # This file
├── prompt_III.ipynb                   # Main analysis notebook
├── data/
│   └── bank-additional-full.csv       # Dataset
└── CRISP-DM-BANK.pdf                 # Research paper
```

### Running the Analysis
1. Ensure all dependencies are installed
2. Place dataset in `data/` folder
3. Open `prompt_III.ipynb` in Jupyter Notebook or JupyterLab
4. Run all cells sequentially

---


