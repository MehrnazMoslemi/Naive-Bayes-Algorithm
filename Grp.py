#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 20:41:39 2022

@author: layss
"""

# data collection 
import pandas as pd 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy.stats import chi2_contingency
import numpy as np




fb_df = pd.read_csv(r'C:\Users\taha\Desktop\mehrnaz.csv')


#data preprocessing

#getting rid of missing values and specializing our sample
print(fb_df.head(5))
print(fb_df.count())
print(fb_df.isnull().sum())







data1=fb_df
data=fb_df

# we reduce our data to look at transfers to the 4 most popular leagues and 
# to look at the 4 positions considered to be the spine of a team
# this will allow us to look at a reduced, specialized, and focused sample
data1=data1[(data1["Position"]=="Goalkeeper") | (data1["Position"]=="Centre-Back") |(data1["Position"]=="Central Midfield")|(data1["Position"]=="Centre-Forward")]
data1=data1[(data1["League_to"]=="Premier League") | (data1["League_to"]=="Ligue 1")|(data1["League_to"]=="LaLiga")|(data1["League_to"]=="Serie A")]
data1=data1[(data1["Season"]=="2018-2019") | (data1["Season"]=="2017-2018")|(data1["Season"]=="2016-2017")|(data1["Season"]=="2015-2016")|(data1["Season"]=="2014-2015")|(data1["Season"]=="2013-2014")]
data1=data1[(data1["League_from"]=="Premier League") | (data1["League_from"]=="Serie A")|(data1["League_from"]=="Ligue 1")|(data1["League_from"]=="LaLiga")|(data1["League_from"]=="1.Bundesliga")|(data1["League_from"]=="SÃ©rie A")]
#2 categorical data
dummy_position = pd.get_dummies(fb_df.Position)
dummy_initial = pd.get_dummies(fb_df.League_to)
dummy_team=pd.get_dummies(fb_df.Team_from)
#dummy_Name=pd.get_dummies(fb_df.Name)



age = fb_df.Age
transfer = fb_df.Transfer_fee



fb_df = fb_df.join(dummy_position)
fb_df = fb_df.join(dummy_initial)
fb_df = fb_df.join(dummy_team)
#fb_df = fb_df.join(dummy_Name)
# 3 getting rid of outliers
low_iqr = fb_df["Transfer_fee"].quantile(0.25) - 1.5* (fb_df["Transfer_fee"].quantile(0.75)-fb_df["Transfer_fee"].quantile(0.25)) 
high_iqr = fb_df["Transfer_fee"].quantile(0.75) + (fb_df["Transfer_fee"].quantile(0.75)-fb_df["Transfer_fee"].quantile(0.25)) *1.5

fb_df = fb_df[(fb_df.Transfer_fee >= low_iqr) |(fb_df.Transfer_fee <= high_iqr)]






# descriptive analysis
data.describe()
print(data.describe())


# Age vs Transfer Fee graphs
sns.catplot(x="Age", y="Transfer_fee", kind="bar", data=data1)
sns.catplot(x="League_from", y="Transfer_fee", kind="bar", data=data1)
sns.catplot(x="Age", y="Transfer_fee", kind="point", data=data1)
sns.catplot(x="League_to", y="Transfer_fee",hue="Age", kind="bar",data=data1)
sns.catplot(x="Position", y="Transfer_fee", hue = "Age", kind="bar", data=data1)
data.plot.scatter(x="Age",y="Transfer_fee")
sns.catplot(x="Season", y="Transfer_fee", kind="point", data=data1)


# Position vs Transfer fee
sns.catplot(x="Position", y="Transfer_fee", kind="bar", data=data1)
sns.catplot(x="Position", y = "Transfer_fee", hue = "League_to",kind = "bar", data = data1)


# Transfer fee vs League of Arrival
sns.catplot(x="League_to", y="Transfer_fee", kind="bar", data=data1)
sns.catplot(x="League_to", y = "Transfer_fee", hue = "Position",kind = "bar", data = data1)


league_from = data.groupby(['League_from'])['Transfer_fee'].sum()
top5sell_league = league_from.sort_values(ascending=False).head(5)
top5sell_league = top5sell_league/1000000
#print(top5sell_league.head())

fig, ax = plt.subplots(figsize=(18,6))
ax.bar(top5sell_league.index, top5sell_league.values, color='green')
ax.set_ylabel("$ millions", color='navy')
ax.set_yticklabels(labels=[i for i in range(0,8000, 1000)], color='navy')
ax.set_xticklabels(labels=top5sell_league.index, color='navy')


league_to = data.groupby(['League_to'])['Transfer_fee'].sum()
top5buy_league = league_to.sort_values(ascending=False).head(5)
top5buy_league = top5buy_league/1000000
#print(top5buy_league.head())
fig, ax = plt.subplots(figsize=(18,6))
ax.bar(top5buy_league.index, top5buy_league.values, color='navy')
ax.set_ylabel("$ millions", color='black')
ax.set_yticklabels(labels=[i for i in range(0,16000, 2000)], color='red')
ax.set_xticklabels(labels=top5buy_league.index, color='red')



diff_league = top5sell_league - top5buy_league
diff_league = diff_league.sort_values(ascending=False)
#print(diff_league.head())
fig, ax = plt.subplots(figsize=(18,6))
ax.bar(diff_league.index, diff_league.values)
ax.set_ylabel("$ millions")

#analysis about clubs
club_from_sum = data.groupby(['Team_from'])['Transfer_fee'].sum()
club_from_count = data.groupby(['Team_from'])['Transfer_fee'].count()
club_from_mean_price = (club_from_sum/1000000) / club_from_count
plt.figure(figsize=(18,6))
sellers_mean = club_from_mean_price.sort_values(ascending=False)[:20]
g = sns.barplot(sellers_mean.index, sellers_mean.values, palette="Reds_r")
g.set_title("Mean price of sold player per club")
g.set(ylabel="$ millions", xlabel="Team selling a player")
plt.xticks(rotation=90)

#######################################################################
club_to_sum = data.groupby(['Team_to'])['Transfer_fee'].sum()
club_to_count = data.groupby(['Team_to'])['Transfer_fee'].count()
club_to_mean_price = (club_to_sum/1000000) / club_to_count

plt.figure(figsize=(18,6))
buy_mean = club_to_mean_price.sort_values(ascending=False)[:20]
g = sns.barplot(buy_mean.index, buy_mean.values, palette=sns.cubehelix_palette(20))
g.set_title("Mean price of bought player per club")
g.set(ylabel="$ millions", xlabel="Team buying a player")
plt.xticks(rotation=90)


######################################################
diff_club = club_from_sum - club_to_sum
diff_club = diff_club.sort_values(ascending=False)
diff_club = diff_club.dropna()

diff_club = diff_club/1000000
diff_club.head(15)

fig, ax = plt.subplots(figsize=(18,6))
make_money = diff_club.sort_values(ascending=False)[:10]
ax.bar(make_money.index, make_money.values, color="pink")
ax.set_title("Clubs that make money on transfer market")
ax.set_ylabel("$ millions")
ax.set_xticklabels(make_money.index, rotation=90)
# ax.autoscale_view()
###################################################

diff_club.tail(15)

fig, ax = plt.subplots(figsize=(18,6))
lose_money = diff_club.sort_values(ascending=True)[:10]
ax.bar(lose_money.index, lose_money.values, color="orange")
ax.set_title("Clubs that lose money on transfer market")
ax.set_ylabel("$ millions")
ax.set_xticklabels(lose_money.index, rotation=90)
ax.autoscale_view()

############################################################



#feature selection part

def cramers_V(var1,var2) :
  
  crosstab =np.array(pd.crosstab(var1,var2, rownames=None, colnames=None)) # Cross table building
  stat = chi2_contingency(crosstab)[0] # Keeping of the test statistic of the Chi2 test
  obs = np.sum(crosstab) # Number of observations
  mini = min(crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table
  return (stat/(obs*mini))


m=cramers_V(fb_df.League_to,fb_df.Season)

print(m)

###################################################


data=fb_df.drop(columns= ["Market_value","League_from","Team_to","Season","Name"])

#Predictive Analysis: Linear Regression
y = data['Transfer_fee']
x = data[["Age"]]

regressor = LinearRegression()
model = regressor.fit(x, y)

print(model.coef_)
print(model.intercept_)
print(model.score(x,y))


# Predictive Analysis: Multiple Linear Regression

y1 = data['Transfer_fee']
x1 =data.drop(columns=['Transfer_fee',"Position","League_to","Team_from"])



x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.30, random_state=42)



regressor1 = LinearRegression()
model1 = regressor1.fit(x1, y1)
y_prediction=regressor1.predict(x1)
score=r2_score(y1,y_prediction)
print("score:",score)
print(model1.coef_)
print(model1.intercept_)




