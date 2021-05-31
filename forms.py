from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField, RadioField
from wtforms.validators import DataRequired, Length, Email, EqualTo


class RegistrationForm(FlaskForm):
	id = StringField('Customer ID', validators=[DataRequired(), Length(min=2, max=20)])
	perc = StringField("Percentage of Premium Paid by Cash", validators=[DataRequired()])
	age = StringField("Age In Days", validators=[DataRequired()])
	income = StringField("Avoid use of commas", validators=[DataRequired()])
	count1 = StringField("Count of 3-6 Months Late", validators=[DataRequired()])
	count2 = StringField("Count of 6-12 Months Late", validators=[DataRequired()])
	count3 = StringField("Count of More Than 12 Months Late", validators=[DataRequired()])
	auc = StringField("Application Underwriting Score", validators=[DataRequired()])
	num = StringField("Number of Premiums Paid", validators=[DataRequired()])
	s_channel = RadioField('Sourcing Channel', choices = ['A', 'B', 'C', 'D', 'E'], validators=[DataRequired()])
	residence = RadioField('Residence Area Type', choices = ['Rural', 'Urban'], validators=[DataRequired()])
	age_group = RadioField('Age Group', choices = ['Teenager', 'Adult', 'Old'], validators=[DataRequired()])
	status = RadioField('Fianancial Status', choices = ['Poor', 'Rich'], validators=[DataRequired()])	
	submit = SubmitField("Predict")