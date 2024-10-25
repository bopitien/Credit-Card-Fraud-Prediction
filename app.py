from flask import Flask, request, render_template, send_file
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

app = Flask(__name__)

# Load the saved model
model = joblib.load('xgb_classifier_model.pkl')

# Define the custom threshold
THRESHOLD = 0.6  # Set this to 0.6 or 0.7 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    # Read the uploaded CSV file into a DataFrame
    try:
        data = pd.read_csv(file)
    except Exception as e:
        return f"Error reading the CSV file: {str(e)}", 400

    # Ensure the data has the correct columns (features expected by the model)
    required_columns = [f'V{i}' for i in range(1, 29)] + ['Amount']

    # Validate that the uploaded file contains all the required columns
    if not all(col in data.columns for col in required_columns):
        return "Uploaded file does not have the required columns", 400

    # Check if 'Class' or 'class' column is present for performance evaluation
    if 'Class' in data.columns:
        actual_labels = data['Class']
        has_actual_labels = True
    elif 'class' in data.columns:
        actual_labels = data['class']
        has_actual_labels = True
    else:
        actual_labels = None
        has_actual_labels = False

    # Select only the relevant features for prediction
    features = data[required_columns]

    # Predict probabilities instead of classes
    probabilities = model.predict_proba(features)[:, 1]  # Get probabilities of the fraud class

    # Apply custom threshold for fraud classification
    predictions = (probabilities >= THRESHOLD).astype(int)

    # Add predictions to the DataFrame
    data['Prediction'] = predictions

    # If actual labels are available, evaluate the model's performance
    if has_actual_labels:
        accuracy = accuracy_score(actual_labels, predictions)
        precision = precision_score(actual_labels, predictions)
        recall = recall_score(actual_labels, predictions)
        f1 = f1_score(actual_labels, predictions)
        roc_auc = roc_auc_score(actual_labels, predictions)

        report = {
            'Accuracy': f"{accuracy:.2f}",
            'Precision': f"{precision:.2f}",
            'Recall': f"{recall:.2f}",
            'F1 Score': f"{f1:.2f}",
            'ROC AUC Score': f"{roc_auc:.2f}"
        }

        # Return the report.html template with the report data
        return render_template('report.html', report=report, download_link='/download')
    else:
        # Save the predictions to a new CSV file
        output_csv = 'predictions.csv'
        data.to_csv(output_csv, index=False)

        # If no actual labels, provide the download link for predictions
        return send_file(output_csv, as_attachment=True)

@app.route('/download')
def download():
    # Serve the predictions CSV file for download
    return send_file('predictions.csv', as_attachment=True)

@app.route('/predict-manual', methods=['POST'])
def predict_manual():
    try:
        # Extract values from the form for all the fields
        features = [
            float(request.form['transaction_type_code'].strip()),
            float(request.form['merchant_code'].strip()),
            float(request.form['transaction_time'].strip()),
            float(request.form['day_of_week'].strip()),
            float(request.form['transaction_location_code'].strip()),
            float(request.form['payment_method_code'].strip()),
            float(request.form['cardholder_age_group'].strip()),
            float(request.form['merchant_category_code'].strip()),
            float(request.form['transaction_amount'].strip()),
            float(request.form['account_balance_before'].strip()),
            float(request.form['account_balance_after'].strip()),
            float(request.form['num_transactions_past_week'].strip()),
            float(request.form['num_transactions_past_month'].strip()),
            float(request.form['avg_transaction_amount_week'].strip()),
            float(request.form['avg_transaction_amount_month'].strip()),
            float(request.form['transaction_currency_code'].strip()),
            float(request.form['currency_exchange_rate'].strip()),
            float(request.form['is_international_transaction'].strip()),
            float(request.form['is_contactless'].strip()),
            float(request.form['num_declines_today'].strip()),
            float(request.form['card_type'].strip()),
            float(request.form['card_issuer_code'].strip()),
            float(request.form['country_issuer_code'].strip()),
            float(request.form['failed_attempts_today'].strip()),
            float(request.form['is_cardholder_verified'].strip()),
            float(request.form['approval_time'].strip()),
            float(request.form['is_online_transaction'].strip()),
            float(request.form['fraud_history'].strip())
        ]

        # Convert to a numpy array and reshape for prediction
        final_features = np.array(features).reshape(1, -1)
        prediction = model.predict(final_features)

        # Map the prediction result
        result = 'Fraudulent Transaction' if prediction[0] == 1 else 'Non-Fraudulent Transaction'

        return render_template('index.html', prediction_text=f'Prediction: {result}')
    except ValueError:
        return "Invalid input. Please ensure all values are numeric.", 400

if __name__ == "__main__":
    app.run(debug=True)
