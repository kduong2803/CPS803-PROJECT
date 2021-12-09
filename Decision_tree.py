#import packages
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn import tree

#read data
data=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
test=test.drop(["PassengerId"])


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
clf=DecisionTreeClassifier(criterion="entropy",max_depth=5)
clf.fit(x_train,y_train)
predict=clf.predict(x_val)

#print accuracy
print(accuracy_score(y_val,predict))

#final prediction
final=clf.predict(test)
df=pd.DataFrame({"PassengerId": testID.values,
                 "Survived":final
                 })
df.to_csv("prediction_j48tree.csv",index=False)




#get feature_name
df_dummified = pd.get_dummies(x)
columns=df_dummified.columns
feature_name=columns.tolist()

#get tree picture
fig = plt.figure(figsize=(25,20))
tree.plot_tree(clf,
                feature_names=feature_name,
                class_names=['0','1'],
                filled=True)

fig.savefig("decision_tree.png")


text_representation = tree.export_text(clf)
print(text_representation)
