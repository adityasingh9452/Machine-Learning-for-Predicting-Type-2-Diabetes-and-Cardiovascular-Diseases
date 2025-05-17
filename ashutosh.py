from flask import Flask, render_template_string, request
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the scaler and models
scaler = joblib.load('scaler.pkl')
models = {
    'Logistic Regression': joblib.load('logistic_regression_model.pkl'),
    'Support Vector Machine': joblib.load('support_vector_machine_model.pkl'),
    'Random Forest': joblib.load('random_forest_model.pkl')
}

# Function to make predictions
def make_predictions(new_data):
    # Scale the new data
    new_data_scaled = scaler.transform(new_data)

    # Make predictions on the new data using all models
    predictions = {}
    for model_name, model in models.items():
        # Predict for Prediabetes
        pred_prediabetes = model.predict(new_data_scaled)
        predictions[model_name] = {'Prediabetes': pred_prediabetes[0]}

        # Predict for Hypertension
        model_hypertension = joblib.load(f'{model_name.replace(" ", "_").lower()}_hypertension_model.pkl')
        pred_hypertension = model_hypertension.predict(new_data_scaled)
        predictions[model_name]['Hypertension'] = pred_hypertension[0]

    return predictions

# HTML templates
index_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diabetes Prediction</title>
</head>
<body>
    <h1>Diabetes Prediction Form</h1>
    <form method="POST">
        <label for="name">Name:</label>
        <input type="text" id="name" name="name" required><br><br>
        
        <label for="insulin">Insulin Level:</label>
        <input type="number" id="insulin" name="insulin" required><br><br>
        
        <label for="glucose">Glucose Level:</label>
        <input type="number" id="glucose" name="glucose" required><br><br>
        
        <label for="blood_pressure">Blood Pressure Level:</label>
        <input type="number" id="blood_pressure" name="blood_pressure" required><br><br>
        
        <input type="submit" value="Submit">
    </form>
</body>
</html>
'''

results_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Results</title>
</head>
<body>
    <h1>Prediction Results for {{ name }}</h1>
    {% for model_name, prediction in predictions.items() %}
        <h2>{{ model_name }}</h2>
        <p>Prediabetes: {{ 'Yes' if prediction['Prediabetes'] == 1 else 'No' }}</p>
        <p>Hypertension: {{ 'Yes' if prediction['Hypertension'] == 1 else 'No' }}</p>
    {% endfor %}
    <a href="/">Go Back</a>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form['name']
        insulin = float(request.form['insulin'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])

        # Only 3 features used to match the scaler
        new_data = pd.DataFrame({
            'Glucose': [glucose],
            'Insulin': [insulin],
            'BloodPressure': [blood_pressure]
        })

        predictions = make_predictions(new_data)
        return render_template_string(results_html, name=name, predictions=predictions)

    return render_template_string(index_html)

if __name__ == '__main__':
    app.run(debug=True)
