import os
import pandas as pd 
import numpy as np 
import flask
import pickle
import joblib
from flask import Flask, render_template, request, flash, redirect
from forms import RegistrationForm
from flask_sqlalchemy import SQLAlchemy

app=Flask(__name__)

app.config["SECRET_KEY"] = "1e6efb89d38a82dc4ca9f4340d9b1e06"


@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
 form = RegistrationForm(csrf_enabled=False)
 if form.validate_on_submit():
      id = int(form.id.data)
      perc = float(form.perc.data)
      age = int(form.age.data)
      income = int(form.income.data)
      c1 = float(form.count1.data)
      c2 = float(form.count2.data)
      c3 = float(form.count3.data)
      auc = float(form.auc.data)
      num = int(form.num.data)
      if form.s_channel.data == "A":
         s_channel = [1,0,0,0,0]
      elif form.s_channel.data == "B":
         s_channel = [0,1,0,0,0]
      elif form.s_channel.data == "C":
         s_channel = [0,0,1,0,0]
      elif form.s_channel.data == "D":
         s_channel = [0,0,0,1,0]
      elif form.s_channel.data == "E":
         s_channel = [0,0,0,0,1]
      if form.residence.data == "Urban":
         residence = [0,1]
      elif form.residence.data == "Rural":
         residence = [1,0]
      if form.age_group.data == "Teenager":
         age_group = [1,0,0]
      elif form.age_group.data == "Adult":
         age_group = [0,1,0]
      elif form.age_group.data == "Old":
         age_group = [0,0,1]
      if form.status.data == "Poor":
         status = [1,0]
      elif form.status.data == "Rich":
         status = [0,1]
      prediction = [id, perc, age, income, c1, c2, c3, auc, num] + s_channel + residence + age_group + status
      prediction = np.array(prediction)
      prediction = prediction.reshape(1, -1)    
      file = open("model.pkl","rb")
      trained_model = joblib.load(file)
      result = trained_model.predict(prediction)
      return str(result)
 return render_template('index.html', title="Home", form=form)


def ValuePredictor(to_predict_list):
 to_predict = np.array(to_predict_list).reshape(1,4)
 loaded_model = pickle.load(open("model.pkl","rb"))
 result = loaded_model.predict(to_predict)
 return result[0]

# @app.route("/predict",methods = ["GET","POST"])
# def result():
#  if request.method == "POST":
#    idd = int(request.form.get('idd'))
#    perc = float(request.form.get('perc'))
#    age = int(request.form.get('age'))
#    income = int(request.form.get('income'))
#    c1 = float(request.form.get('count1'))
#    c2 = float(request.form.get('count2'))
#    c3 = float(request.form.get('count3'))
#    auc = float(request.form.get('auc'))
#    num = int(request.form.get('num'))
#    s_channel = list(request.form.get('s_channel'))
#    residence = list(request.form.get('residence'))
#    age_group = list(request.form.get('age_group'))
#    status = list(request.form.get('status'))

if __name__ == "__main__":
 app.run(debug=True)