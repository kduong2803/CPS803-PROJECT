#import packages
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')



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
# Initialising the NN
# layers
L1=20
L2=20
L3=5
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(L1,input_shape=(x_train.shape[1],),kernel_regularizer='l2', activation='relu'))
model.add(tf.keras.layers.Dense(L2,kernel_regularizer='l2', activation='relu'))
model.add(tf.keras.layers.Dense(L3,kernel_regularizer='l2', activation='relu'))
model.add(tf.keras.layers.Dense(1,kernel_regularizer='l2', activation='sigmoid'))


# Compiling the NN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the NN
l=model.fit(x_train, y_train, batch_size = 32, epochs = 200)


#print accuracy
preds = model.evaluate(x=x_val, y=y_val)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
print ("Training Accuracy = " + str(l.history['accuracy'][-1]))

#final prediction
y_pred = model.predict(test)
y_final = (y_pred > 0.5).astype(int).reshape(test.shape[0])
df=pd.DataFrame({"PassengerId": testID.values,
                 "Survived":y_final
                 })
df.to_csv("neural_network.csv",index=False)





