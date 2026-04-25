from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import os

app = Flask(__name__)

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'LinearRegression.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'train.csv')

# ---------------- LOAD MODEL ----------------
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print("Model loading error:", e)
    model = None

# ---------------- LOAD DATA ----------------
try:
    car = pd.read_csv(DATA_PATH)
except Exception as e:
    print("Dataset loading error:", e)
    car = pd.DataFrame()  # fallback to avoid crash


# ---------------- HOME + PREDICT ----------------
@app.route('/', methods=['GET', 'POST'])
def index():

    Cylinders = sorted(car['Cylinders'].unique()) if not car.empty else []
    Fuel_Type = sorted(car['Fuel_Type'].unique()) if not car.empty else []
    Road_Type = sorted(car['Road_Type'].unique()) if not car.empty else []
    Age_years = sorted(car['Age_years'].unique()) if not car.empty else []

    prediction = None

    if request.method == 'POST':
        try:
            Cylinders_val = int(request.form.get('Cylinders'))
            Fuel_Type_val = request.form.get('Fuel_Type')
            Road_Type_val = request.form.get('Road_Type')
            Age_years_val = float(request.form.get('Age_years'))
            Weight_kg = float(request.form.get('Weight_kg'))
            Horsepower = float(request.form.get('Horsepower'))

            # Create input DataFrame
            input_data = pd.DataFrame(
                [[Weight_kg, Horsepower, Cylinders_val, Fuel_Type_val, Road_Type_val, Age_years_val]],
                columns=[
                    'Weight_kg',
                    'Horsepower',
                    'Cylinders',
                    'Fuel_Type',
                    'Road_Type',
                    'Age_years'
                ]
            )

            # Predict
            if model:
                prediction = round(model.predict(input_data)[0], 2)
            else:
                prediction = "Model not loaded"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template(
        'index.html',
        Cylinders=Cylinders,
        Fuel_Type=Fuel_Type,
        Road_Type=Road_Type,
        Age_years=Age_years,
        prediction=prediction
    )


# ---------------- RUN APP ----------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render dynamic port
    app.run(host='0.0.0.0', port=port)
