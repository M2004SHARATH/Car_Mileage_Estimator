from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load model
try:
    model = pickle.load(open('LinearRegression.pkl', 'rb'))
except Exception as e:
    print("Model loading error:", e)
    model = None

# Load dataset
car = pd.read_csv('train.csv')


@app.route('/', methods=['GET', 'POST'])
def index():
    Cylinders = sorted(car['Cylinders'].unique())
    Fuel_Type = sorted(car['Fuel_Type'].unique())
    Road_Type = sorted(car['Road_Type'].unique())
    Age_years = sorted(car['Age_years'].unique())

    prediction = None

    if request.method == 'POST':
        try:
            Cylinders_val = int(request.form.get('Cylinders'))
            Fuel_Type_val = request.form.get('Fuel_Type')
            Road_Type_val = request.form.get('Road_Type')
            Age_years_val = float(request.form.get('Age_years'))
            Weight_kg = float(request.form.get('Weight_kg'))
            Horsepower = float(request.form.get('Horsepower'))

            input_data = pd.DataFrame(
                [[Weight_kg, Horsepower, Cylinders_val, Fuel_Type_val, Road_Type_val, Age_years_val]],
                columns=['Weight_kg', 'Horsepower', 'Cylinders', 'Fuel_Type', 'Road_Type', 'Age_years']
            )

            prediction = np.round(model.predict(input_data)[0], 2)

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


if __name__ == '__main__':
    app.run(debug=True)