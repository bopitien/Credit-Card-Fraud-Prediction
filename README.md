# **Credit Card Fraud Detection System**

This project is a **Credit Card Fraud Detection System** built using **Flask**. It leverages an **XGBoost Classifier** for fraud detection, incorporating a custom threshold to balance **precision** and **recall** effectively. The application provides a responsive, user-friendly web interface for both batch and manual fraud predictions.

---

## **Features**

- **Batch Prediction**: Upload a CSV file with transaction data, and the model classifies each entry as either fraudulent or non-fraudulent. If the file includes actual labels (`Class` or `class`), it also generates a detailed **performance report** with metrics such as accuracy, precision, recall, F1 score, and ROC AUC score.
  
- **Manual Prediction**: Submit transaction details manually to receive an instant fraud classification.

- **Adjustable Threshold**: Configure a **custom threshold** to improve either precision or recall, depending on model requirements.

- **CSV Download**: Download the predictions as a **CSV file** for further analysis.

---

## **Project Overview**

The model in this project was trained using anonymized **credit card transaction data** to detect fraudulent activities effectively.Applying a custom threshold, the model emphasizes **precision**, aiming to minimize false positives and enhance fraud detection accuracy.

---

## **Model**

The project uses an **XGBoost Classifier**, which was trained on both **balanced** and **imbalanced** datasets. This adaptability enables the model to handle real-world usage scenarios where fraud cases are typically rare, optimizing both precision and recall depending on the threshold.

---

## **Dataset**

The dataset used for training contains over **550,000 records** of anonymized European credit card transactions from **2023**. It includes feature columns labeled **V1 to V28** and a binary **class** column, where `1` indicates fraud and `0` indicates non-fraud transactions.

---

## **App link**
https://credit-fraud-detecting-app-5df6dbd8d3f3.herokuapp.com/


