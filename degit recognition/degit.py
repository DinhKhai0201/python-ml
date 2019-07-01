import pandas as pd
import numpy 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
dataset = pd.read_csv('train.csv')
X_train = dataset.iloc[:40000,1:]
y_train = dataset.iloc[:40000,0]
clf =RandomForestClassifier(n_estimators =100)
clf.fit(X_train,y_train)
X_test = dataset.iloc[40000:,1:]
y_test = dataset.iloc[40000:,0]
predict = clf.predict(X_test)
print(accuracy_score(y_test,predict))
#----test
test = pd.read_csv('test.csv')
predict = clf.predict(test)
a = numpy.asarray(predict).reshape(28000,1)
b = numpy.arange(1,28001).reshape(28000,1)
result = numpy.hstack((b,a))
print(result)
pd.DataFrame(result).to_csv('result',header=['ImageId','Label'],index =None)

