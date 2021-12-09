#import packages
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt

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

#transforming string data to number
label=preprocessing.LabelEncoder()
cols=["Sex","Embarked"]
for i in cols:
    data[i]=label.fit_transform(data[i])
    test[i]=label.transform(test[i])

#split data to train
y= data["Survived"]
x=data.drop(["Survived"],axis=1)
x_train,x_val,y_train,y_val= train_test_split(x,y,test_size=0.2, random_state=50)

#model selection
clf=LogisticRegression()
clf.fit(x_train,y_train)
predict=clf.predict(x_val)

#print accuracy
print(accuracy_score(y_val,predict))

#final prediction
final=clf.predict(test)
df=pd.DataFrame({"PassengerId": testID.values,
                 "Survived":final
                 })
df.to_csv("prediction_logisticregression.csv",index=False)

fpr, tpr, _ = metrics.roc_curve(y_val,  predict)
auc = metrics.roc_auc_score(y_val,  predict)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.savefig('ROC_rate.png')
plt.show()