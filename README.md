
# 🚢 Titanic Survival Prediction

This project builds a Decision Tree Classifier to predict survival outcomes for passengers aboard the Titanic, using a cleaned version of the Titanic dataset from Kaggle.

## 📌 Project Overview

The goal is to apply data preprocessing, exploratory data analysis, and machine learning techniques to predict whether a passenger survived or not, based on features like age, gender, class, fare, and embarkation port.

---

## 🧰 Tools & Libraries

- Python 3
- pandas
- numpy
- seaborn, matplotlib
- scikit-learn

---

## 📁 Files

- `Improved_Titanic_Decision_Tree_Model.ipynb` – Final notebook with improved structure, visualizations, and evaluation.
- `titanic.csv` – Dataset used for training and testing.
- `README.md` – Project overview and instructions.

---

## 🚀 How to Run

1. Clone the repository or download the files.
2. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Open the notebook:
   ```bash
   jupyter notebook Improved_Titanic_Decision_Tree_Model.ipynb
   ```

---

## 📊 Key Steps in the Notebook

- **Data Cleaning**: Handle missing values and drop irrelevant features.
- **Encoding**: Convert categorical features to numerical values.
- **EDA**: Generate visual insights using plots and heatmaps.
- **Model Training**: Use Decision Tree with controlled depth and leaf size.
- **Evaluation**: Check accuracy, classification report, and confusion matrix.
- **Feature Importance**: Identify which variables contributed most to predictions.

---

## 📈 Results

- **Model**: Decision Tree Classifier
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score
- **Feature Importance**: `Sex`, `Fare`, `Pclass` were most predictive.

---

## 🔍 Future Improvements

- Try different models: Random Forest, Logistic Regression, or SVM.
- Use GridSearchCV for hyperparameter tuning.
- Implement cross-validation for more robust evaluation.
- Add additional feature engineering for better performance.
