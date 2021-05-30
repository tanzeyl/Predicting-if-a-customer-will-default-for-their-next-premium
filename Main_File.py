# %%
"""
<a href="https://colab.research.google.com/github/tanzeyl/Predicting-if-a-customer-will-default-for-their-next-premium/blob/main/Main_File.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

# %%
"""
# Step 1: Problem Statement: To predict if a customer will pay their premium on time or not.
"""

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
"""
## Step 2: Hypothesis Generation:
Following can be the factors that can be used to predict if a customer will pay their premium on time:
1. Whether previous premium is payed.
2. Time of previous payment
3. Type of job
4. Area of residence

## Step 3: Data Extraction:
Files have been provided beforehand.

## Step 4: Data Exploration:
Given below:
"""

# %%
#Importing important modules
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# %%
#Reading the data
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Predicting-if-a-customer-will-default-for-their-next-premium/train.csv")

# %%
#Displaying first five rows
data.head()

# %%
#Checking number of rows and columns
data.shape

# %%
#Displaying names of columns
data.columns

# %%
"""
## Variable Identification:
Dependent Variable: Target
Independent Variable: perc_premium_paid_by_cash_credit', age_in_days, Income, Count_3-6_months_late, Count_6-12_months_late, Count_more_than_12_months_late, application_underwriting_score, no_of_premiums_paid, sourcing_channel, residence_area_type
"""

# %%
data.dtypes

# %%
"""
Here, sourcing_channel, residence_area_type are categorical variables and rest others are continuous variables.
"""

# %%
"""
# Univariate Analysis For Continuous Variables
"""

# %%
data.describe()

# %%
"""
Here we can see that count_3-6_months_late, count_6-12_months_late, count_more_than_12_months_late, and application_underwriting_score have missing values. We will fill these missing values later in this segment. First, we will draw histograms and box-plots for each independent continuous variable to see their distributions and check if they have outliers.
"""

# %%
data["perc_premium_paid_by_cash_credit"].plot.hist()

# %%
"""
Here, we can see that a majority of people have not paid their premium through cash, but with some other means. 
"""

# %%
data["perc_premium_paid_by_cash_credit"].plot.box()

# %%
"""
We will be using mean to find the central tendency of columns with no outliers and median for columns with outliers.
"""

# %%
print(data["perc_premium_paid_by_cash_credit"].mean())

# %%
"""
This means that on an average, people have paid 31.42% of their premium in cash.
"""

# %%
data["age_in_days"].plot.hist()

# %%
"""
Here, we can see a majority of our customers lie in the age range of 54-68 years (20,000-25,000 days).
"""

# %%
data["age_in_days"].plot.box()

# %%
"""
### Here, we can see that "age_in_days" has outliers. Let's make a separate list of all columns having outliers for future ease.
"""

# %%
outliers_list = []
outliers_list.append("age_in_days")
outliers_list

# %%
print(data["age_in_days"].min())
print(data["age_in_days"].max())
print(data["age_in_days"].median())

# %%
"""
Our youngest customer is 7670 days (approx. 21 years), oldest customer is of 37602 days (approx. 103 years), and the average age of our customers is 18625 days (approx. 51 years).
"""

# %%
data["Income"].plot.hist()

# %%
data["Income"].plot.box()

# %%
outliers_list.append("Income")
outliers_list

# %%
print(data["Income"].min())
print(data["Income"].max())
print(data["Income"].median())

# %%
"""
The lowest income of among our customers is 24,030 Rs. while the highest being 9,02,62,600 RS with an average of 1,66,560 Rs. Note that customers with lower income are less likely to pay premium on time.
"""

# %%
data["application_underwriting_score"].plot.hist()

# %%
"""
This is a left-skewed distribution telling us that approximately 40,000 customers have a good (near 100) application underwriting score.
"""

# %%
data["application_underwriting_score"].plot.box()

# %%
outliers_list.append("application_underwriting_score")
outliers_list

# %%
print(data["application_underwriting_score"].min())
print(data["application_underwriting_score"].max())
print(data["application_underwriting_score"].median())

# %%
"""
Maximum A.U.S. = 99.89

Minimum A.U.S. = 91.9

Average A.U.S. = 99.21
"""

# %%
data["no_of_premiums_paid"].plot.hist()

# %%
"""
This is a right skewed distribution telling us that approximately 35,000 customers have paid atleast 10 premiums. Note that the number of customers decrease as the number of premiums paid increases, which means they both have a negative correlation.

"""

# %%
data["no_of_premiums_paid"].plot.box()

# %%
outliers_list.append("no_of_premiums_paid")
outliers_list

# %%
print(data["no_of_premiums_paid"].min())
print(data["no_of_premiums_paid"].max())
print(data["no_of_premiums_paid"].median())

# %%
"""
Lease number of premiums paid by a customer = 2

Most number of premiums paid by a customer = 60

Average premiums paid = 10
"""

# %%
data["target"].plot.hist()

# %%
"""
This shows that a high number of customers are likely to pay their premiums on time.
"""

# %%
data["target"].plot.box()

# %%
"""
# Unvariate Analysis of Categorical Variables
"""

# %%
"""
Here, I have treated Count_3-6_months_late, Count_6-12_months_late, and Count_more_than_12_months_late as categorical variables as there is not much variation in their values.
"""

# %%
data["Count_3-6_months_late"].value_counts()

# %%
data["Count_3-6_months_late"].value_counts().plot.bar()

# %%
data["Count_6-12_months_late"].value_counts()

# %%
data["Count_6-12_months_late"].value_counts().plot.bar()

# %%
data["Count_more_than_12_months_late"].value_counts()

# %%
data["Count_more_than_12_months_late"].value_counts().plot.bar()

# %%
"""
Here, we can see that usually alot of our customers have paid their premiums on time.
"""

# %%
data["sourcing_channel"].value_counts()

# %%
data["sourcing_channel"].value_counts().plot.bar()

# %%
data["residence_area_type"].value_counts()

# %%
data["residence_area_type"].value_counts().plot.bar()

# %%
"""
# Bivariate Analysis
### Our target variable here is continuous, hence we will perform continuous-continuous B.A. and continuous-categorical B.A.
"""

# %%
"""
## Proceeding with continuous-continuous bivariate analysis 
"""

# %%
data.corr()

# %%
data.plot.scatter("target", "perc_premium_paid_by_cash_credit")

# %%
data["target"].corr(data["perc_premium_paid_by_cash_credit"])

# %%
data.plot.scatter("target", "age_in_days")

# %%
data["target"].corr(data["age_in_days"])

# %%
data.plot.scatter("target", "Income")

# %%
data["target"].corr(data["Income"])

# %%
data.plot.scatter("target", "application_underwriting_score")

# %%
data["target"].corr(data["application_underwriting_score"])

# %%
data.plot.scatter("target", "no_of_premiums_paid")

# %%
data["target"].corr(data["no_of_premiums_paid"])

# %%
"""
### Continuous-Categorical Bivariate Analysis

"""

# %%
data.groupby("sourcing_channel")["target"].mean()

# %%
data.groupby("sourcing_channel")["target"].mean().plot.bar()

# %%
"""
Here, it is evident that if a customer will pay their premium or not is not much affected by the sourcing channel they use.
"""

# %%
data.groupby("residence_area_type")["target"].mean()

# %%
data.groupby("residence_area_type")["target"].mean().plot.bar()

# %%
data.groupby("Count_3-6_months_late")["target"].mean()

# %%
data.groupby("Count_3-6_months_late")["target"].mean().plot.bar()

# %%
data.groupby("Count_6-12_months_late")["target"].mean()

# %%
data.groupby("Count_6-12_months_late")["target"].mean().plot.bar()

# %%
data.groupby("Count_more_than_12_months_late")["target"].mean()

# %%
data.groupby("Count_more_than_12_months_late")["target"].mean().plot.bar()

# %%
"""
# Missng Value Treatment
"""

# %%
data.isnull().sum()

# %%
"""
### Here, we can see that Count_3-6_months_late, Count_6-12_months_late, Count_more_than_12_months_late have 97 missing values and application_underwriting_score has 2974 missing values.
"""

# %%
"""
Since we are treating the Count_3-6_months_late, Count_6-12_months_late, and Count_more_than_12_months_late as categorical values, we will be filling their missing values using mode, while application_writing_underscore's missing values will be filled by it's mean
"""

# %%
def null(df):
    data['application_underwriting_score'].fillna(data['application_underwriting_score'].mean(),inplace=True)
    data['Count_3-6_months_late'].fillna(0,inplace=True)
    data['Count_6-12_months_late'].fillna(0,inplace=True)
    data['Count_more_than_12_months_late'].fillna(0,inplace=True)
    return df

# %%
null(data)

# %%
data.isnull().sum()

# %%
"""
We can see that no column has anymore missing values.
"""

# %%
"""
# Univariate Outlier Detection.
"""

# %%
"""
Let's recall our list containing all columns containing outliers.
"""

# %%
outliers_list

# %%
"""
Let's start by calculating quantiles and IQRs for each column having outliers.
"""

# %%
q1 = int(data.age_in_days.quantile([0.25]))
q3 = int(data.age_in_days.quantile([0.75]))
IQR = q3 - q1
upper_limit = int(q3+ 1.5 * IQR)
lower_limit = int(q1 - 1.5 * IQR)
print("Upper limit is {} and lower limit is {}.".format(upper_limit,lower_limit))

# %%
data.loc[data["age_in_days"]>upper_limit, "age_in_days"] = np.mean(data["age_in_days"])

# %%
data.loc[data["age_in_days"]<lower_limit, "age_in_days"] = np.mean(data["age_in_days"])

# %%
data["age_in_days"].plot.box()

# %%
q1 = int(data.Income.quantile(0.25))
q3 = int(data.Income.quantile(0.75))
IQR = q3 - q1
upper_limit = int(q3+ 1.5 * IQR)
lower_limit = int(q1 - 1.5 * IQR)
print("Upper limit is {} and lower limit is {}.".format(upper_limit,lower_limit))

# %%
data.loc[data["Income"]>upper_limit, "Income"] = np.mean(data["Income"])

# %%
data.loc[data["Income"]<lower_limit, "Income"] = np.mean(data["Income"])

# %%
data["Income"].plot.box()

# %%
q1 = int(data.application_underwriting_score.quantile(0.25))
q3 = int(data.application_underwriting_score.quantile(0.75))
IQR = q3 - q1
upper_limit = int(q3+ 1.5 * IQR)
lower_limit = int(q1 - 1.5 * IQR)
print("Upper limit is {} and lower limit is {}.".format(upper_limit,lower_limit))

# %%
"""
Looking at the consistency of data in the column "application_underwriting_score", I think it is better to leave it like this.
"""

# %%
q1 = int(data.no_of_premiums_paid.quantile(0.25))
q3 = int(data.no_of_premiums_paid.quantile(0.75))
IQR = q3 - q1
upper_limit = int(q3+ 1.5 * IQR)
lower_limit = int(q1 - 1.5 * IQR)
print("Upper limit is {} and lower limit is {}.".format(upper_limit,lower_limit))

# %%
data.loc[data["no_of_premiums_paid"]>upper_limit, "no_of_premiums_paid"] = np.mean(data["no_of_premiums_paid"])

# %%
data.loc[data["no_of_premiums_paid"]<lower_limit, "no_of_premiums_paid"] = np.mean(data["no_of_premiums_paid"])

# %%
data["no_of_premiums_paid"].plot.box()

# %%
"""
We still have two outlying data points. Let's fix that by setting upper-limit = 22.
"""

# %%
data.loc[data["no_of_premiums_paid"]>22 ,"no_of_premiums_paid"] = np.mean(data["no_of_premiums_paid"])

# %%
data["no_of_premiums_paid"].plot.box()

# %%
"""
# Variable Transformation
"""

# %%
data["age_in_days"].plot.hist()

# %%
np.log(data["age_in_days"]).plot.hist()

# %%
"""
Log transformation gives us a more symmteric curve.
"""

# %%
data["age_in_days"].plot.hist()

# %%
bins = [0, 5475, 18250,33960]
group = ["Teenager", "Adult", "Old"]

# %%
data["Age Group"] = pd.cut(data["age_in_days"], bins, labels=group)

# %%
data["Age Group"].value_counts()

# %%
bins = [0, 100000, 468140]
group = ["Poor", "Rich"]
data["Fianancial Status"] = pd.cut(data["Income"], bins, labels=group)

# %%
data["Fianancial Status"].value_counts()

# %%
data.head()

# %%
"""
# Model Building
"""

# %%
"""
Since, we have a dependent variable, we will be using supervised learning models and also our dependent variable is categorical, therefore we will be using supervised learning classification models.
"""

# %%
"""
Let's read our test data set first.
"""

# %%
test = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Predicting-if-a-customer-will-default-for-their-next-premium/test.csv")

# %%
data = pd.get_dummies(data)

# %%
"""
Now, let's divide our data set as per the requirement. We will go on the top to import required modules.
"""

# %%
train, test = train_test_split(data, test_size=0.2)

# %%
x_train=train.drop('target',axis=1)
y_train=train['target']
x_test=test.drop('target',axis=1)
y_test=test['target']

# %%
"""
# KNN Model
Let's go up and import KNN.
We will be doing this for five different values of K i.e. 1, 2, 3, 4 and 5.

"""

# %%
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)
performance = metrics.accuracy_score(y_test, predictions)
print(performance)

# %%
"""
## When K = 1, accuracy is about 88.6%
"""

# %%
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)
performance = metrics.accuracy_score(y_test, predictions)
print(performance)

# %%
"""
## When K = 2, accuracy is about 83.8 %.  So K = 1 is better option till now.
"""

# %%
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)
performance = metrics.accuracy_score(y_test, predictions)
print(performance)

# %%
"""
## 92.7%. Not bad. 
"""

# %%
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)
performance = metrics.accuracy_score(y_test, predictions)
print(performance)

# %%
"""
## Good but we have had better.
"""

# %%
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)
performance = metrics.accuracy_score(y_test, predictions)
print(performance)

# %%
"""
K = 5 will be our final decision which helps our model get an accuracy of 93.5%.
"""

# %%
"""
# Model Deployment
"""

# %%
import pickle
filename = 'model.pkl'
pickle.dump(knn, open(filename, 'wb'))

# %%
