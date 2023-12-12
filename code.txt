import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv('E:\\ddos\\kdd.csv')
x=dataset.drop(columns=list(dataset.columns)[-1])
y=dataset[list(dataset.columns)[-1]]
x
le= preprocessing.LabelEncoder()
x['protocol_type']=le.fit_transform(x['protocol_type'])
x['service']=le.fit_transform(x['service'])
x['flag']=le.fit_transform(x['flag'])
y=le.fit_transform(y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
dp=pd.DataFrame(x_train)
dp
dp.info()
nor=preprocessing.Normalizer()
x_train=nor.fit_transform(x_train)
x_test=nor.fit_transform(x_test)
dk=pd.DataFrame(x_train)
dk
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
clf = RandomForestClassifier(n_estimators=50)
clf.fit(x_train,y_train)
pred=clf.predict(x_test)
#accuracy_score(y_test,pred)
np.mean(y_test==pred)
sorted(np.c_[range(len(x.columns)),x.columns,clf.feature_importances_],reverse=True,key=lambda x:x[2])
alpha=0.05
pred_vars=[]
max_feature=0.0
best_feature=""
for feature in zip(range(len(x)),x.columns,clf.feature_importances_):
    if feature[2]>float(alpha):
        pred_vars.append(feature[0])
        if max_feature<feature[2]:
            best_feature=feature[1]
            max_feature=feature[2]
clfk=RandomForestClassifier(n_estimators=50)
clfk.fit(x_train[:,pred_vars],y_train)
predictions=clfk.predict(x_test[:,pred_vars])
np.mean(predictions==y_test)
x_train[:,pred_vars].shape
print(best_feature)
from sklearn.model_selection import cross_val_score
scores=cross_val_score(clf,x_train,y_train,cv=5)
np.mean(scores)


code with two seperate datasets taken:
=============================
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split;
from sklearn.svm import SVC
trainsRows = [] 
testRows=[]
alpha=0.22;
t=0.00000000005;
A_index=[];
B_index=[];
# reading csv file
trainFilename="KDDTrain+.csv";
testFilename="KDDTest+.csv";
train=pd.read_csv("E:\\New usecase\\"+trainFilename);
test=pd.read_csv("E:\\New usecase\\"+testFilename);
x_train=train.drop(columns=list(train.columns)[-1])
x_test=train[list(train.columns)[-1]]
y_train=test.drop(columns=list(test.columns)[-1])
y_test=test[list(test.columns)[-1]]
le= preprocessing.LabelEncoder()
x_train['protocol_type']=le.fit_transform(x_train['protocol_type'])
x_train['service']=le.fit_transform(x_train['service'])
x_train['flag']=le.fit_transform(x_train['flag'])
y_train['protocol_type']=le.fit_transform(y_train['protocol_type'])
y_train['service']=le.fit_transform(y_train['service'])
y_train['flag']=le.fit_transform(y_train['flag'])
x_test=le.fit_transform(x_test)
y_test=le.fit_transform(y_test)

clf = RandomForestClassifier(n_estimators=50)
clf.fit(x_train,x_test)
pred=clf.predict(y_train)
#accuracy_score(y_test,pred)
np.mean(y_test==pred)

code with svm:
==========
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv('E:\\New usecase\\kdd.csv')
x=dataset.drop(columns=list(dataset.columns)[-1])
y=dataset[list(dataset.columns)[-1]]
x
le= preprocessing.LabelEncoder()
x['protocol_type']=le.fit_transform(x['protocol_type'])
x['service']=le.fit_transform(x['service'])
x['flag']=le.fit_transform(x['flag'])
y=le.fit_transform(y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

nor=preprocessing.Normalizer()
x_train=nor.fit_transform(x_train)
x_test=nor.fit_transform(x_test)

from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
clf = RandomForestClassifier(n_estimators=50)
clf.fit(x_train,y_train)
pred=clf.predict(x_test)
#accuracy_score(y_test,pred)
np.mean(y_test==pred)




t=0.00000000005;
alpha=0.05
A_index=[]
B_index=[]
max_feature=0.0
best_feature=0
for feature in zip(range(len(x)),x.columns,clf.feature_importances_):
    if feature[2]>float(alpha):
        A_index.append(feature[0])
        if max_feature<feature[2]:
            best_feature=feature[0]
            max_feature=feature[2]
    else:
        B_index.append(feature[0]);


sum1=0.00;
for xi in B_index:
    predict_a=[str(best_feature)]
    clfA = SVC(kernel='linear',C=1.0,gamma=0.2,cache_size=7000)  
    clfB = SVC(kernel='linear',C=1.0,gamma=0.2,cache_size=7000)  
    print("SVC initialization done");
    clfA.fit(x_train[A_index],y_train)
    print("SVM on A is done........")
    clfB.fit(x_train[A_index],y_train)
    print("SVM on B is done........")
    predictA=clfA.predict(x_test[best_feature]);
    predictB=clfB.predict(x_test[best_feature]);
    for feature in zip(best_feature, clfA.feature_importances_):
        print(feature)
        print("\n");
        sum1=sum1+feature[1];
    for feature in zip(xi, clfB.feature_importances_):
        print(feature)
        sum1=sum1-feature[1];
    mean_sum=float(sum1/len(x_train));
    if mean_sum>t:
        A_index.append(xi); 
