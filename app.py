# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 13:10:10 2021

@author: Varshitha Sonalika
"""

from flask import Flask, render_template, request
import pickle
import numpy as np

filename = 'C:/Users/Varshitha Sonalika/Desktop/diabetes prediction/app.py'
LR = pickle.load(open(filename,'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('dbcode.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['Pregnancies'])
        glucose = int(request.form['Glucose'])
        bp = int(request.form['BloodPressure'])
        skin = int(request.form['SkinThickness'])
        bmi = float(request.form['BMI'])
        insulin = int(request.form['Insulin'])
        dpf = float(request.form['DiabetesPedigreeFunction'])
        age = int(request.form['Age'])
        data = np.array([[preg, glucose, bp, skin, bmi, insulin, dpf, age]])
        my_prediction = LR.predict(data)
        
        return render_template('dbcode.html', prediction = my_prediction)
    if __name__=='__main__':
        app.run(port = 5000, threaded=True)
        
