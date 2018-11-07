
## Question 3: Purchasing Insurance


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')
%matplotlib inline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import seaborn as sns
sns.set()

Caravan_train = pd.read_csv("http://www.webpages.uidaho.edu/~stevel/Datasets/Caravan_train.csv")
Caravan_train.head(1)
```


```python
Caravan_test = pd.read_csv("http://www.webpages.uidaho.edu/~stevel/Datasets/Caravan_unk.csv")
Caravan_test.head(1)
```


```python
Caravan_train['Purchase'] = Caravan_train.Purchase.map({'No':0,'Yes':1})
```


```python
X=Caravan_train.drop('Purchase',axis=1)
y=Caravan_train.Purchase
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)
```

Check for missing values


```python
Caravan_train[Caravan_train.isnull().any(axis=1)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MOSTYPE</th>
      <th>MAANTHUI</th>
      <th>MGEMOMV</th>
      <th>MGEMLEEF</th>
      <th>MOSHOOFD</th>
      <th>MGODRK</th>
      <th>MGODPR</th>
      <th>MGODOV</th>
      <th>MGODGE</th>
      <th>MRELGE</th>
      <th>...</th>
      <th>APERSONG</th>
      <th>AGEZONG</th>
      <th>AWAOREG</th>
      <th>ABRAND</th>
      <th>AZEILPL</th>
      <th>APLEZIER</th>
      <th>AFIETS</th>
      <th>AINBOED</th>
      <th>ABYSTAND</th>
      <th>Purchase</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 86 columns</p>
</div>



# Model Testing
I will use different classification models to predict whether the customer profile will buy insurance

First model is Logistic Regression


```python
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))
```

    Accuracy of Logistic regression classifier on training set: 0.94
    Accuracy of Logistic regression classifier on test set: 0.93
    


```python
y_pred_class = logreg.predict(X_test)
y_pred_prob = logreg.predict_proba(X_test)[:, 1]
print ('Accuracy Score: {:.5f}'.format(metrics.accuracy_score(y_test, y_pred_class)))
print ('AUC: {:.5f}'.format(metrics.roc_auc_score(y_test, y_pred_prob)))
print metrics.confusion_matrix(y_test, y_pred_class)
```

    Accuracy Score: 0.92919
    AUC: 0.70791
    [[1076    3]
     [  79    0]]
    


```python
from sklearn import metrics, cross_validation
logreg=LogisticRegression()
predicted = cross_validation.cross_val_predict(logreg, X, y, cv=10)
print metrics.accuracy_score(y, predicted)
print metrics.classification_report(y, predicted) 
```

    0.9369330453563715
                 precision    recall  f1-score   support
    
              0       0.94      1.00      0.97      4346
              1       0.25      0.01      0.03       284
    
    avg / total       0.90      0.94      0.91      4630
    
    

Plot ROC curve for Logisitic Regression


```python
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
```




    Text(0,0.5,'True Positive Rate (Sensitivity)')




![png](output_12_1.png)


Next, Nearest Neighbor.  I will try n_neighbors of 1 and 50


```python
from sklearn.neighbors import KNeighborsClassifier
nn = KNeighborsClassifier(n_neighbors=1)
nn.fit(X_train, y_train)
print("Test set R^2: {:.2f}".format(nn.score(X_test, y_test)))
```

    Test set R^2: 0.89
    


```python
y_pred_class = nn.predict(X_test)
y_pred_prob = nn.predict_proba(X_test)[:, 1]
print ('Accuracy Score: {:.5f}'.format(metrics.accuracy_score(y_test, y_pred_class)))
print ('AUC: {:.5f}'.format(metrics.roc_auc_score(y_test, y_pred_prob)))
print metrics.confusion_matrix(y_test, y_pred_class)
```

    Accuracy Score: 0.89119
    AUC: 0.54274
    [[1021   58]
     [  68   11]]
    


```python
from sklearn.neighbors import KNeighborsClassifier
nn = KNeighborsClassifier(n_neighbors=50)
nn.fit(X_train, y_train)
print("Test set R^2: {:.2f}".format(nn.score(X_test, y_test)))
```

    Test set R^2: 0.93
    


```python
y_pred_class = nn.predict(X_test)
y_pred_prob = nn.predict_proba(X_test)[:, 1]
print ('Accuracy Score: {:.5f}'.format(metrics.accuracy_score(y_test, y_pred_class)))
print ('AUC: {:.5f}'.format(metrics.roc_auc_score(y_test, y_pred_prob)))
print metrics.confusion_matrix(y_test, y_pred_class)
```

    Accuracy Score: 0.93178
    AUC: 0.70672
    [[1079    0]
     [  79    0]]
    

A range of n values will be tested and plotted to determine best accuracy


```python
k_range = range(1,50)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
```




    Text(0,0.5,'Cross-Validated Accuracy')




![png](output_19_1.png)


From testng a range of n values in the KNN model, it appears that accuracy isn't enhanced with any more neighbors above 13.
Cross validation will be used to determine the optimal n_neighbors


```python
from sklearn.model_selection import cross_val_score
myList = list(range(1,20))
neighbors = filter(lambda x: x % 2 != 0, myList)
cv_scores = []
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
```


```python

MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print "The optimal number of neighbors is %d" % optimal_k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
```

    The optimal number of neighbors is 11
    


![png](output_22_1.png)


Cross Validation provided an optimal n neighbors of 13. I will use this number in my model


```python
from sklearn.neighbors import KNeighborsClassifier
nn = KNeighborsClassifier(n_neighbors=13)
nn.fit(X_train, y_train)
print("Test set R^2: {:.2f}".format(nn.score(X_test, y_test)))
```

    Test set R^2: 0.93
    


```python
y_pred_class = nn.predict(X_test)
y_pred_prob = nn.predict_proba(X_test)[:, 1]
print ('Accuracy Score: {:.5f}'.format(metrics.accuracy_score(y_test, y_pred_class)))
print ('AUC: {:.5f}'.format(metrics.roc_auc_score(y_test, y_pred_prob)))
print metrics.confusion_matrix(y_test, y_pred_class)
```

    Accuracy Score: 0.93092
    AUC: 0.70141
    [[1078    1]
     [  79    0]]
    

Plot ROC curve for K Nearest Neighbor


```python
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
```




    Text(0,0.5,'True Positive Rate (Sensitivity)')




![png](output_27_1.png)


Next models to try are the Decision Tree Classifer and the Random Forest Classifier


```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
```

    Accuracy of Decision Tree classifier on training set: 0.99
    Accuracy of Decision Tree classifier on test set: 0.88
    


```python
y_pred_class = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:, 1]
print ('Accuracy Score: {:.5f}'.format(metrics.accuracy_score(y_test, y_pred_class)))
print ('AUC: {:.5f}'.format(metrics.roc_auc_score(y_test, y_pred_prob)))
print metrics.confusion_matrix(y_test, y_pred_class)
```

    Accuracy Score: 0.88428
    AUC: 0.51603
    [[1016   63]
     [  71    8]]
    


```python
from sklearn.ensemble import RandomForestClassifier
fclf=RandomForestClassifier().fit(X_train,y_train)
print('Accuracy of Decision Forest classifier on training set: {:.2f}'
     .format(fclf.score(X_train, y_train)))
print('Accuracy of Decision Forest classifier on test set: {:.2f}'
     .format(fclf.score(X_test, y_test)))
```

    Accuracy of Decision Forest classifier on training set: 0.98
    Accuracy of Decision Forest classifier on test set: 0.92
    


```python
y_pred_class = fclf.predict(X_test)
y_pred_prob = fclf.predict_proba(X_test)[:, 1]
print ('Accuracy Score: {:.5f}'.format(metrics.accuracy_score(y_test, y_pred_class)))
print ('AUC: {:.5f}'.format(metrics.roc_auc_score(y_test, y_pred_prob)))
print metrics.confusion_matrix(y_test, y_pred_class)
```

    Accuracy Score: 0.91710
    AUC: 0.60960
    [[1059   20]
     [  76    3]]
    

The next model is Naive Bayes
I will run a MultinomialNB and the GaussianNB for comparison.


```python
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)
print('Accuracy of Naive Bayes classifier on training set: {:.2f}'
     .format(nb.score(X_train, y_train)))
print('Accuracy of Naive Bayes classifier on test set: {:.2f}'
     .format(nb.score(X_test, y_test)))
```

    Accuracy of Naive Bayes classifier on training set: 0.77
    Accuracy of Naive Bayes classifier on test set: 0.75
    


```python
y_pred_class = nb.predict(X_test)
y_pred_prob = nb.predict_proba(X_test)[:, 1]
print metrics.accuracy_score(y_test, y_pred_class)
print metrics.roc_auc_score(y_test, y_pred_prob)
print metrics.confusion_matrix(y_test, y_pred_class)
```

    0.7452504317789291
    0.716404077849861
    [[821 258]
     [ 37  42]]
    


```python
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy of Naive Bayes classifier on training set: {:.2f}'
     .format(gnb.score(X_train, y_train)))
print('Accuracy of Naive Bayes classifier on test set: {:.2f}'
     .format(gnb.score(X_test, y_test)))
```

    Accuracy of Naive Bayes classifier on training set: 0.12
    Accuracy of Naive Bayes classifier on test set: 0.12
    


```python
y_pred_class = gnb.predict(X_test)
y_pred_prob = gnb.predict_proba(X_test)[:, 1]
print metrics.accuracy_score(y_test, y_pred_class)
print metrics.roc_auc_score(y_test, y_pred_prob)
print metrics.confusion_matrix(y_test, y_pred_class)
```

    0.1234887737478411
    0.7079809012095118
    [[  67 1012]
     [   3   76]]
    

Gaussian is best suited for data with a normal distribution. With majority false positives in the confusion matrix, GaussianNB is not appropriate for this data set.

Caravan data will now be run on a Neural Network model


```python
from sklearn.neural_network import MLPClassifier
mlpc = MLPClassifier(solver='lbfgs')
mlpc.fit(X_train,y_train)
print('Accuracy of NN classifier on training set: {:.2f}'
     .format(mlpc.score(X_train, y_train)))
print('Accuracy of NN classifier on test set: {:.2f}'
     .format(mlpc.score(X_test, y_test)))
```

    Accuracy of NN classifier on training set: 0.96
    Accuracy of NN classifier on test set: 0.92
    


```python
y_pred_class = mlpc.predict(X_test)
y_pred_prob = mlpc.predict_proba(X_test)[:, 1]
print metrics.accuracy_score(y_test, y_pred_class)
print metrics.roc_auc_score(y_test, y_pred_prob)
print metrics.confusion_matrix(y_test, y_pred_class)
```

    0.9153713298791019
    0.626179604261796
    [[1056   39]
     [  59    4]]
    

The final model to test will be the Support Vector Machine.


```python
from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,y_train)
print('Accuracy of SVC classifier on training set: {:.2f}'
     .format(svc.score(X_train, y_train)))
print('Accuracy of SVC classifier on test set: {:.2f}'
     .format(svc.score(X_test, y_test)))
```

    Accuracy of SVC classifier on training set: 0.94
    Accuracy of SVC classifier on test set: 0.95
    


```python
y_pred_class = svc.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)
print metrics.confusion_matrix(y_test, y_pred_class)
```

    0.9455958549222798
    [[1095    0]
     [  63    0]]
    

# Final step will be to apply my most accuract model and get the top 50 customer profiles to approach about buying insurance


```python
Caravan_test = pd.read_csv("http://www.webpages.uidaho.edu/~stevel/Datasets/Caravan_unk.csv")
```

I will apply the logistic regression classifer to the Caravan test data.  This model had the highest area under curve score.


```python
y_preds=logreg.predict_proba(Caravan_test)[:,1]
Caravan_test["preds"]=y_preds
Caravan_test.nlargest(50,'preds')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MOSTYPE</th>
      <th>MAANTHUI</th>
      <th>MGEMOMV</th>
      <th>MGEMLEEF</th>
      <th>MOSHOOFD</th>
      <th>MGODRK</th>
      <th>MGODPR</th>
      <th>MGODOV</th>
      <th>MGODGE</th>
      <th>MRELGE</th>
      <th>...</th>
      <th>APERSONG</th>
      <th>AGEZONG</th>
      <th>AWAOREG</th>
      <th>ABRAND</th>
      <th>AZEILPL</th>
      <th>APLEZIER</th>
      <th>AFIETS</th>
      <th>AINBOED</th>
      <th>ABYSTAND</th>
      <th>preds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30</th>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.908511</td>
    </tr>
    <tr>
      <th>557</th>
      <td>30</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.651081</td>
    </tr>
    <tr>
      <th>738</th>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.595645</td>
    </tr>
    <tr>
      <th>651</th>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.470672</td>
    </tr>
    <tr>
      <th>868</th>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>2</td>
      <td>2</td>
      <td>8</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.414618</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.406065</td>
    </tr>
    <tr>
      <th>389</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0.381937</td>
    </tr>
    <tr>
      <th>267</th>
      <td>33</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>8</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.372782</td>
    </tr>
    <tr>
      <th>693</th>
      <td>7</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>9</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.370382</td>
    </tr>
    <tr>
      <th>739</th>
      <td>39</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>9</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.353890</td>
    </tr>
    <tr>
      <th>439</th>
      <td>22</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>9</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.342150</td>
    </tr>
    <tr>
      <th>156</th>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.325129</td>
    </tr>
    <tr>
      <th>852</th>
      <td>7</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>8</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.321919</td>
    </tr>
    <tr>
      <th>471</th>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.321638</td>
    </tr>
    <tr>
      <th>31</th>
      <td>8</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.319594</td>
    </tr>
    <tr>
      <th>29</th>
      <td>6</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
      <td>9</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.312449</td>
    </tr>
    <tr>
      <th>482</th>
      <td>33</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
      <td>9</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.306810</td>
    </tr>
    <tr>
      <th>423</th>
      <td>6</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.288003</td>
    </tr>
    <tr>
      <th>414</th>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.285170</td>
    </tr>
    <tr>
      <th>41</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>7</td>
      <td>9</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.284242</td>
    </tr>
    <tr>
      <th>56</th>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.283339</td>
    </tr>
    <tr>
      <th>217</th>
      <td>8</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.277985</td>
    </tr>
    <tr>
      <th>1059</th>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.277813</td>
    </tr>
    <tr>
      <th>420</th>
      <td>13</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.273046</td>
    </tr>
    <tr>
      <th>403</th>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>8</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.272809</td>
    </tr>
    <tr>
      <th>639</th>
      <td>36</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.263637</td>
    </tr>
    <tr>
      <th>579</th>
      <td>33</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>8</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>9</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.257237</td>
    </tr>
    <tr>
      <th>630</th>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.254513</td>
    </tr>
    <tr>
      <th>1004</th>
      <td>8</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.249184</td>
    </tr>
    <tr>
      <th>257</th>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>8</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.248945</td>
    </tr>
    <tr>
      <th>319</th>
      <td>12</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>7</td>
      <td>9</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.247888</td>
    </tr>
    <tr>
      <th>920</th>
      <td>36</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>8</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.246643</td>
    </tr>
    <tr>
      <th>20</th>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.244351</td>
    </tr>
    <tr>
      <th>32</th>
      <td>33</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.243608</td>
    </tr>
    <tr>
      <th>754</th>
      <td>36</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.240351</td>
    </tr>
    <tr>
      <th>1045</th>
      <td>13</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.239172</td>
    </tr>
    <tr>
      <th>681</th>
      <td>13</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.238938</td>
    </tr>
    <tr>
      <th>164</th>
      <td>32</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.237227</td>
    </tr>
    <tr>
      <th>11</th>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.232545</td>
    </tr>
    <tr>
      <th>26</th>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.232545</td>
    </tr>
    <tr>
      <th>836</th>
      <td>22</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.228522</td>
    </tr>
    <tr>
      <th>457</th>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.228217</td>
    </tr>
    <tr>
      <th>201</th>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>3</td>
      <td>9</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.227287</td>
    </tr>
    <tr>
      <th>703</th>
      <td>36</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.225971</td>
    </tr>
    <tr>
      <th>518</th>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>9</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.224801</td>
    </tr>
    <tr>
      <th>19</th>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>8</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.223976</td>
    </tr>
    <tr>
      <th>849</th>
      <td>13</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.223640</td>
    </tr>
    <tr>
      <th>487</th>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>9</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.217793</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>7</td>
      <td>9</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.217782</td>
    </tr>
    <tr>
      <th>644</th>
      <td>6</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.217316</td>
    </tr>
  </tbody>
</table>
<p>50 rows × 86 columns</p>
</div>


