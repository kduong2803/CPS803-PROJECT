#package
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


#read data
data=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
testID=test["PassengerId"]

#clean data
def clean(data):
    data=data.drop(["Ticket","Cabin","Name","PassengerId"],axis=1)
    columns=["SibSp","Parch","Fare","Age"]
    for i in columns:
        data[i].fillna(data[i].median(),inplace=True)
    data.Embarked.fillna("U",inplace=True)
    return data

data=clean(data)
test=clean(test)



#split data to train
y= data["Survived"]
x=data.drop(["Survived"],axis=1)
x_train,x_val,y_train,y_val= train_test_split(x,y,test_size=0.2, random_state=50)

#Overall Survived rate
data['Survived'].value_counts().plot.pie(autopct = '%1.2f%%');

#Sex
data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar();

#Class
data[['Pclass', 'Survived']].groupby('Pclass').mean().plot.bar();

#Embarked
plt.show()

#Age
age_data =data[['Age', 'Survived']]
sns.histplot(age_data['Age'])

ageFacet=sns.FacetGrid(data,hue='Survived',aspect=3)
ageFacet.map(sns.kdeplot,'Age',shade=True)
ageFacet.set(xlim=(0,data['Age'].max()))
ageFacet.add_legend()

# Fare
fareFacet=sns.FacetGrid(data,hue='Survived',aspect=3)
fareFacet.map(sns.kdeplot,'Fare',shade=True)
fareFacet.set(xlim=(0,150))
fareFacet.add_legend()
plt.show()
# Embarked

embarkedBar=sns.barplot(data=data[data.Embarked!="U"],x='Embarked',y='Survived')
plt.show()
#SibSp
sibspFacet=sns.FacetGrid(data,hue='Survived',aspect=3)
sibspFacet.map(sns.kdeplot,'SibSp',shade=True)
sibspFacet.set(xlim=(0,7))
sibspFacet.add_legend()
plt.show()

#Parch
ParchFacet=sns.FacetGrid(data,hue='Survived',aspect=3)
ParchFacet.map(sns.kdeplot,'Parch',shade=True)
ParchFacet.set(xlim=(0,7))
ParchFacet.add_legend()
plt.show()



