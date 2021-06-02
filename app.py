import os
import pandas as pd 
import numpy as np 
import flask
import pickle
import joblib
from flask import Flask, render_template, request
import requests

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
 return render_template('index.html')

@app.route("/predict",methods = ["GET","POST"])
def result():
 if request.method == "POST":
   idd = int(request.form.get('idd'))
   perc = float(request.form.get('perc'))
   age = int(request.form.get('age'))
   income = int(request.form.get('income'))
   c1 = float(request.form.get('count1'))
   c2 = float(request.form.get('count2'))
   c3 = float(request.form.get('count3'))
   auc = float(request.form.get('auc'))
   num = int(request.form.get('num'))
   s_channel = list(request.form.get('s_channel'))
   residence = list(request.form.get('residence'))
   age_group = list(request.form.get('age_group'))
   status = list(request.form.get('status'))
   prediction = [idd, perc, age, income, c1, c2, c3, auc, num] + s_channel + residence + age_group + status
   prediction = np.array(prediction)
   prediction = prediction.reshape(1, -1)    
   file = open("model.pkl","rb")
   trained_model = joblib.load(file)
   result = trained_model.predict(prediction)
   return str(result[0])

if __name__ == "__main__":
 app.run(debug=True)