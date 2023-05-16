import pandas as pd
import numpy as np
import pickle
import time
from flask import Flask, render_template, request
import requests
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from flask import Flask, send_file
import io

app = Flask(__name__)
model = pickle.load(open('heart_2020_cleaned (1).pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/heart')
def heart_disease_data():
    return render_template('heart_disease_data.html')

@app.route('/prediction', methods = ["POST", "GET"])
def heart_disease_predictioon():
    if request.method=='POST':
        BMI = float(request.form["BMI"])
        PhysicalHealth = float(request.form["PhysicalHealth"])
        MentalHealth = float(request.form["MentalHealth"])
        SleepTime = float(request.form["SleepTime"])
        ans=model.predict(np.array([[BMI, PhysicalHealth, MentalHealth, SleepTime ]]))
        result=ans[0]
        return render_template('prediction.html', result = result)
    return render_template('heart_disease_prediction.html')


@app.route('/url_query')
def contact_request():
    name = request.args.get('name')
    email = request.args.get('email')
    return f'{name} and {email}'

@app.route('/contact', methods = ["GET", "POST"])
def contact_page():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        return render_template('submitted.html', name = name, email = email)
    return render_template('contact.html')



if __name__ == '__main__':
    app.run(debug = True)