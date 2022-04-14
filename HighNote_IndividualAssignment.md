## Highnote       



```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
```


```python
data = pd.read_csv('/Users/michelleskaf/Documents/MSBA2022/DataMining/HN_data_PostModule.csv')
data = data.iloc[:,1:]
```


```python
X = data.iloc[:,:24].values
y = data.iloc[:,-1].values

```


```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X)
X=imputer.transform(X)
```


```python
#splitting the dataset to 75/25 train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
```


```python
#feature scaling the data 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
```


```python
#pip install xgboost
#import xgboost as xgb
```


```python
#create a function for a model 
def models(X_train, y_train):
        
        #KNN
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        
        #Logistic Regression
        from sklearn.linear_model import LogisticRegression
        log = LogisticRegression(random_state=0)
        log.fit(X_train, y_train)
        
        #Decision Tree
        from sklearn.tree import DecisionTreeClassifier
        tree = DecisionTreeClassifier(random_state=0, criterion='entropy')
        tree.fit(X_train, y_train)
        
        #Random Forest 
        from sklearn.ensemble import RandomForestClassifier
        forest = RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=0)
        forest.fit(X_train, y_train)
        
        #XGBoost
        #from xgb
        
        #Print accuracy on train data 
        print('[0]KNN Accuracy:', knn.score(X_train, y_train))
        print('[1]Logistic Regression Accuracy:', log.score(X_train, y_train))
        print('[2]Decision Tree Accuracy:', tree.score(X_train, y_train))
        print('[3]Random Forest Accuracy:', forest.score(X_train, y_train))
        
        return knn, log, tree, forest
```


```python
#accuracy on the training data
model = models(X_train, y_train)
```

    [0]KNN Accuracy: 0.937656232511286
    [1]Logistic Regression Accuracy: 0.9319603526968374
    [2]Decision Tree Accuracy: 1.0
    [3]Random Forest Accuracy: 0.9884963126018231



```python
#test model accuracy on test data
from sklearn.metrics import confusion_matrix

for i in range(len(model)):
    print('Model', i)
    cm = confusion_matrix(y_test,model[i].predict(X_test))

    TP = cm[0][0]
    TN = cm[1][1]
    FP = cm[1][0]
    FN = cm[0][1]

    print(cm)
    print(('Testing Accuracy:', (TP + TN)/(TP + TN + FP + FN)))
    print()

```

    Model 0
    [[24783   189]
     [ 1749    83]]
    ('Testing Accuracy:', 0.9276973586031936)
    
    Model 1
    [[24822   150]
     [ 1754    78]]
    ('Testing Accuracy:', 0.92896582599612)
    
    Model 2
    [[20939  4033]
     [ 1416   416]]
    ('Testing Accuracy:', 0.7967094463512908)
    
    Model 3
    [[24809   163]
     [ 1738    94]]
    ('Testing Accuracy:', 0.9290777495896135)
    



```python
from sklearn.metrics import classification_report, accuracy_score

for i in range(len(model)):
    print('Model', i)
    print(classification_report(y_test, model[i].predict(X_test)))
    print(accuracy_score(y_test, model[i].predict(X_test)))
    print()
```

    Model 0
                  precision    recall  f1-score   support
    
               0       0.93      0.99      0.96     24972
               1       0.31      0.05      0.08      1832
    
        accuracy                           0.93     26804
       macro avg       0.62      0.52      0.52     26804
    weighted avg       0.89      0.93      0.90     26804
    
    0.9276973586031936
    
    Model 1
                  precision    recall  f1-score   support
    
               0       0.93      0.99      0.96     24972
               1       0.34      0.04      0.08      1832
    
        accuracy                           0.93     26804
       macro avg       0.64      0.52      0.52     26804
    weighted avg       0.89      0.93      0.90     26804
    
    0.92896582599612
    
    Model 2
                  precision    recall  f1-score   support
    
               0       0.94      0.84      0.88     24972
               1       0.09      0.23      0.13      1832
    
        accuracy                           0.80     26804
       macro avg       0.52      0.53      0.51     26804
    weighted avg       0.88      0.80      0.83     26804
    
    0.7967094463512908
    
    Model 3
                  precision    recall  f1-score   support
    
               0       0.93      0.99      0.96     24972
               1       0.37      0.05      0.09      1832
    
        accuracy                           0.93     26804
       macro avg       0.65      0.52      0.53     26804
    weighted avg       0.90      0.93      0.90     26804
    
    0.9290777495896135
    



```python

```
