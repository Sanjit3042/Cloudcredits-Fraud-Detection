# Fraud Detection System

## Project Overview
This project implements a **Fraud Detection System** to identify fraudulent transactions in financial data. The system is built using various **classification algorithms** and **anomaly detection techniques** to detect suspicious activities in credit card transactions. The project also includes visualization to show the distribution of normal and fraudulent transactions.

## Features
- **Classification Models**: Logistic Regression, Random Forest.
- **Anomaly Detection**: Isolation Forest for unsupervised fraud detection.
- **Data Preprocessing**: Handling of class imbalance, feature scaling, and data splitting.
- **Evaluation Metrics**: Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix.
- **Visualizations**: Fraud detection visualizations, ROC curves, confusion matrices.

## Dataset
This project uses the **Credit Card Fraud Detection Dataset**, which is publicly available on Kaggle.

- [Download the Credit Card Fraud Dataset from Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)

### Dataset Summary
The dataset contains credit card transactions made by European cardholders over two days. The dataset is highly imbalanced, with most transactions being legitimate.

- **Instances**: 284,807
- **Fraudulent Transactions**: 492 (0.172%)
- **Classes**: 
  - `0`: Normal transactions
  - `1`: Fraudulent transactions

### Files
- `creditcard.csv`: Contains transaction data with features and labels.

## Project Structure
```plaintext
fraud_detection/
│
├── data/                         # Directory to store the dataset (not included in GitHub)
│   └── creditcard.csv             # Download this file manually
├── main.py                       # Main script to train and evaluate models
├── fraud_detection.py             # Core fraud detection functions
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation (this file)
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/fraud_detection.git
   cd fraud_detection
   ```

2. **Install the required packages**:
   Use `pip` to install the necessary Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   - Download the dataset from Kaggle: [Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).
   - Place the `creditcard.csv` file in the `data/` directory.

4. **Run the project**:
   To train the models and visualize the results, run the `main.py` script:
   ```bash
   python main.py
   ```

## Usage

### 1. Data Preprocessing
The dataset is preprocessed by:
- Scaling features using `StandardScaler`.
- Handling imbalanced classes using techniques like **class weights** and **undersampling**.

### 2. Model Training
The project uses two main models for detecting fraud:
- **Logistic Regression**: A simple, interpretable classification algorithm.
- **Random Forest**: A more powerful ensemble model for improving accuracy.

### 3. Anomaly Detection
Anomaly detection is performed using **Isolation Forest**, which is particularly useful for identifying rare fraudulent transactions.

### 4. Visualization
The project generates various visualizations to help analyze model performance:
- **Confusion Matrix**: Shows true positives, true negatives, false positives, and false negatives.
- **ROC Curves**: Plots the trade-off between true positive rate and false positive rate for each model.
- **Fraud Distribution**: Displays the number of normal vs. fraudulent transactions in the dataset.

## Evaluation Metrics
Given the imbalanced nature of the dataset, the following evaluation metrics are used to measure model performance:
- **Precision**: Fraction of predicted frauds that were actually fraud.
- **Recall**: Fraction of actual frauds that were detected.
- **F1-Score**: Harmonic mean of Precision and Recall.
- **AUC-ROC**: Area under the ROC curve, showing the ability of the model to distinguish between fraud and normal transactions.

## Example Visualizations

### Confusion Matrix:
![Confusion Matrix](confusion_matrix.png)

### ROC Curve:
![ROC Curve](roc_curve.png)

## Next Steps and Improvements
- Implement **SMOTE** (Synthetic Minority Over-sampling Technique) to handle the class imbalance more effectively.
- Experiment with more advanced models such as **XGBoost** or **Neural Networks**.
- Add additional feature engineering techniques to extract more meaningful features from the transaction data.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Acknowledgments
- The dataset used for this project is provided by **Kaggle** and can be found [here](https://www.kaggle.com/mlg-ulb/creditcardfraud).
- Libraries used: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.
