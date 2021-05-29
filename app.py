import os
import pandas as pd 
import numpy as np 
import flask
import pickle
import joblib
from flask import Flask, render_template, request
app=Flask(__name__)
@app.route("/")
def index():
 return flask.render_template("index.html")
def ValuePredictor(to_predict_list):
 to_predict = np.array(to_predict_list).reshape(1,4)
 loaded_model = pickle.load(open("model.pkl","rb"))
 result = loaded_model.predict(to_predict)
 return result[0]
@app.route("/predict",methods = ["POST"])
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
   try:
      prediction = np.array(prediction)
      prediction = prediction.reshape(1, -1)    
      file = open("model.pkl","rb")
      trained_model = joblib.load(file)
      result = trained_model.predict(prediction)
      return str(result)
   except Exception:
      return Exception.msg()
if __name__ == "__main__":
 app.run(debug=True)