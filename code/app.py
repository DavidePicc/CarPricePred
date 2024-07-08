from flask import Flask, render_template, request
from joblib import load
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        km_driven = float(request.form['km_driven'])
        year = float(request.form['year'])
        mark = request.form['mark']
        name = request.form['name']

        df = pd.read_csv("/Applications/XAMPP/xamppfiles/htdocs/PEagle/flask/our_db_5.csv")


        scaler_km = load('/Applications/XAMPP/xamppfiles/htdocs/PEagle/flask/def_scaler_km.pkl')
        scaler_year = load('/Applications/XAMPP/xamppfiles/htdocs/PEagle/flask/def_scaler_year.pkl')
        model = load_model('/Applications/XAMPP/xamppfiles/htdocs/PEagle/flask/def_model.keras')
        
        
        marks =sorted(df['mark'].unique())
        names =sorted(df['name'].unique())
        mark_dummies=pd.get_dummies(marks, columns='mark')
        name_dummies=pd.get_dummies(names, columns='name')


        if mark in marks:
            mark_pred = np.array(mark_dummies.get(mark))
        else:
            return print('Marchio non esistente')

        if name in names:
            name_pred = np.array(name_dummies.get(name))
        else:
            return print('Modello non esistente')


        normalized_km = scaler_km.transform([[km_driven]])
        normalized_year = scaler_year.transform([[year]])

        user_input = np.hstack((normalized_km,normalized_year, mark_pred.reshape(1,-1), name_pred.reshape(1,-1)))
        predicted_price = model.predict(user_input)
        print('Il prezzo predetto per la macchina è: ' + str(predicted_price[0][0]))



        return render_template('result.html', predicted_price=predicted_price[0][0])

if __name__ == '__main__':
    app.run(debug=True)
