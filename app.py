import os
import pandas as pd 
import numpy as np 
import flask
import pickle
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
   perc = request.form.get('perc')
   age = request.form.get('age')
   income = request.form.get('income')
   c1 = request.form.get('c1')
   c2 = request.form.get('c2')
   c3 = request.form.get('c3')
   auc = request.form.get('auc')
   num = request.form.get('num')
   s_channel = request.form.get('s_channel')
   residence = request.form.get('residence')
   prediction = [perc, age, income, c1, c2, c3, auc, num, s_channel, residence]
 return render_template("predict.html",prediction=prediction)
if __name__ == "__main__":
 app.run(debug=True)