###################################################
###  MARKETING ANALYTICS                        ###
###   Gülhan Damla Aşık 2000136                 ###
###   Midterm Project - Car Data Classification ###
###################################################

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split ,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance

import os
os.getcwd()
os.chdir("C:/Users/user/Desktop/BAU Lessons/2- Marketing Analysis/Project/Car Data Project")

cols =["buying", "maint", "doors", "persons","lug_boot",  "safety" , "target"]
CarDF0 = pd.read_csv("car_data.csv", sep=",", index_col=False)
CarDF0.columns = cols

CarDF0.info()
CarDF0.head()
CarDF0.describe()
CarDF0.isnull().sum()
# Categorical, no null value

CarDF0.buying.value_counts()
CarDF0.maint.value_counts()
CarDF0.doors.value_counts()
CarDF0.persons.value_counts()
CarDF0.lug_boot.value_counts()
CarDF0.safety.value_counts()
CarDF0.target.value_counts(normalize=True)
# unacc    1209
# acc       384
# good       69
# vgood      65
# All values are equally distributed. Target is not.

labels = ['acc', 'good', 'unacc', 'vgood']
colors = ['skyblue', 'lightsteelblue', 'cadetblue', 'cornflowerblue']
size = [384, 69, 1210, 65]
explode = [0.1, 0.1, 0.1, 0.1]
plt.rcParams['figure.figsize'] = (10, 10)
plt.pie(size, labels = labels, colors = colors, explode = explode, shadow = True, autopct = "%.2f%%")
plt.title('A Pie Chart Representing target Distribution', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()


CarDF0.groupby("target").buying.value_counts()
# buying ="low,medium" only occurs when target is good or vgood.
CarDF0.groupby("target").persons.value_counts()
# 2 persons only occur when target is unacc.
CarDF0.groupby("target").safety.value_counts()
# all target=vgood samples has high safety.

fig, ax = plt.subplots(3,2,figsize=(26,20))
buyingprice = pd.crosstab(CarDF0['buying'], CarDF0['target'])
buyingprice.div(buyingprice.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, ax=ax[0,0])
buyingprice = pd.crosstab(CarDF0['maint'], CarDF0['target'])
buyingprice.div(buyingprice.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, ax=ax[1,0])
buyingprice = pd.crosstab(CarDF0['doors'], CarDF0['target'])
buyingprice.div(buyingprice.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, ax=ax[2,0])
buyingprice = pd.crosstab(CarDF0['persons'], CarDF0['target'])
buyingprice.div(buyingprice.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, ax=ax[0,1])
buyingprice = pd.crosstab(CarDF0['safety'], CarDF0['target'])
buyingprice.div(buyingprice.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, ax=ax[1,1])
buyingprice = pd.crosstab(CarDF0['lug_boot'], CarDF0['target'])
buyingprice.div(buyingprice.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, ax=ax[2,1])

CarDF = CarDF0.copy()
CarDF2 = CarDF0.copy()
CarDF2.target.replace(("unacc", "acc", "good", "vgood"), ("bad", "bad", "good", "good"), inplace = True)
CarDF['target'].value_counts()
CarDF2['target'].value_counts()
# I will replace my target with new values and decrease the classes into 2 in CarDF2.

# Label Encoding
label_encoder = preprocessing.LabelEncoder()
CarDF= CarDF.apply(preprocessing.LabelEncoder().fit_transform)
CarDF2= CarDF2.apply(preprocessing.LabelEncoder().fit_transform)
CarDF2.head()

# split dataset into train/test
X = CarDF.drop(["target"], axis = 1)
y = CarDF.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state = 109, stratify=y)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

CarDF0.target.value_counts()
# unacc    1209     2
# acc       384     0
# good       69     1
# vgood      65     3

CarDF.target.value_counts()
# 2    1209
# 0     384
# 1      69
# 3      65


#######################
#######################
#######################  Model 1 : KNN  ###
#######################
#######################
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
predictionknn= knn.predict(X_test)
KNNTrainScore = knn.score(X_train, y_train)
KNNTestScore = knn.score(X_test, y_test)
print("Training Accuracy: ",KNNTrainScore)
print("Testing Accuracy: ", KNNTestScore)
# Training Accuracy:  0.9710264900662252
# Testing Accuracy:  0.9210019267822736

confusionknn = confusion_matrix(y_test, predictionknn)
print("Confusion Matrix\n")
print(confusionknn)
print(classification_report(y_test, predictionknn))
# class 0 is acc: prediction looks ok
# class 1 is good: out of 21 good labeled cars, 11 of them true and 10 of them wrong. Recall is low, so this is not very good prediction.
# class 2 is unacc: it has most labels and prediction numbers are high, good prediction.
# class 3 is vgood: it has lowest number of variables and prediction looks ok.
y_test.value_counts()


######## try with standardization
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_trainstand = sc.fit_transform(X_train)
X_teststand = sc.transform(X_test)
KNNTrainScoreStand = knn.score(X_trainstand, y_train)
KNNTestScoreStand = knn.score(X_teststand, y_test)
knn.fit(X_trainstand, y_train)
predictionknnstandart = knn.predict(X_teststand)
print("Training Accuracy: ", KNNTrainScoreStand)
print("Testing Accuracy: ", KNNTestScoreStand)
# Training Accuracy:  0.9718543046357616
# Testing Accuracy:  0.9036608863198459
# Train accuracy increased 0,0008. No need for standardization.
# It would be usefull if the data was numerical and some numbers were big, some number were small.


# find the optimal n_neighbors
neighbors = np.arange(1, 15)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
a = np.empty()
# Loop over different values of k
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
# n_neighbors=7 looks like optimal spot. 
# In the plot, right is underfit, model gets less complex and 
# desicions boundries are smoother. The left is opposite.

##### Feature importance
results = permutation_importance(knn, X, y, scoring='neg_mean_squared_error')
importance = results.importances_mean
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
plt.bar([x for x in range(len(importance))], importance)
plt.show()
# Feature: 0, Score: 0.43752
# Feature: 1, Score: 0.37035
# Feature: 2, Score: 0.08083
# Feature: 3, Score: 0.41204
# Feature: 4, Score: 0.21783
# Feature: 5, Score: 0.56213
# Most important feature for KNN is safety with .56, second buying price and persons.


#######################
#######################
#######################  Model 2 : Logistic Regression  ###
#######################
####################### 
lr = LogisticRegression(C=1, penalty="l1")
# C is regularization. When C is low, model coefs are low and model is more regule and training accuracy is low.
# penaly is kind of logistic regression. l1 = Lasso also makes feature selection. l2= Ridge, default setting.
parameters = {"C": [0.01, 0.1, 1, 10], 
              "penalty": ["l1", "l2"]}
searcher = GridSearchCV(lr, parameters)
searcher.fit(X_train, y_train)      # multi class label
lrscore = searcher.best_score_
print("Tuned Logistic Regression Parameters: {}".format(searcher.best_params_)) 
print("Best score is {}".format(searcher.best_score_))
# Tuned Logistic Regression Parameters: {'C': 0.01, 'penalty': 'l2'}
# Best score is 0.7028188333733411

############################################################### Lr with 2 class target
# # I labeled my target as good and bad only.
X2 = CarDF2.drop(["target"], axis = 1)
y2 = CarDF2.iloc[:, -1]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,y2,test_size = 0.3,random_state = 109, stratify=y)
print (X_train2.shape, y_train2.shape)
print (X_test2.shape, y_test2.shape)
CarDF2.target.value_counts()
# bad 1593  0
# good 134  1
y_test2.value_counts()
lr = LogisticRegression(C=0.01)
lr.fit(X_train2, y_train2)
predictionlr = lr.predict(X_test2)
LRTrainScore = lr.score(X_train2, y_train2)
LRTestScore = lr.score(X_test2, y_test2)
print("Training Accuracy: ",LRTrainScore)
print("Testing Accuracy: ", LRTestScore)
# Training Accuracy:  0.9230132450331126
# Testing Accuracy:  0.9210019267822736
# Accuracy increased significantly.

confusionlr = confusion_matrix(y_test2, predictionlr)
print("Confusion Matrix\n")
print(confusionlr)
# From the confusion matrix, it looks like model predicted all labels as 0.
# Even if the accuracy is 0,92, not a good model. 
print(classification_report(y_test2, predictionlr))

#######################
#######################
#######################  Model 3 : Decision Tree  ###
#######################
####################### 
dtgini = DecisionTreeClassifier(max_depth=8, criterion="gini", random_state = 100)   
dtgini.fit(X_train, y_train)   # with multiclass
predictiondtgini = dtgini.predict(X_test)
DTScoregini = accuracy_score(y_test, predictiondtgini)
print("Accuracy with gini: ",DTScoregini)
# Accuracy with gini:  0.9344894026974951

dtentropy = DecisionTreeClassifier(max_depth=8, criterion="entropy", random_state = 100)
dtentropy.fit(X_train, y_train)  # with multiclass
predictiondtentropy = dtentropy.predict(X_test)
DTScoreentropy = accuracy_score(y_test, predictiondtentropy)
print("Accuracy with entropy for multi class: ",DTScoreentropy)
# Accuracy with entropy for multi class:  0.9402697495183044

dtentropy.fit(X_train2, y_train2)  # with binary class
predictiondtentropy2 = dtentropy.predict(X_test2)
DTScoreentropy2 = accuracy_score(y_test2, predictiondtentropy2)
print("Accuracy with entropy for binary class: ",DTScoreentropy2)
# Accuracy with entropy for binary class:  0.9749518304431599

confusiondt = confusion_matrix(y_test2, predictiondtentropy2)
print("Confusion Matrix\n")
print(confusiondt)
# Prediction looks fine.
print(classification_report(y_test2, predictiondtentropy2))


##### Hyperparameter tuning for binary class
params_dt = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],'criterion': ["gini", "entropy"]}
# 5-fold stratified cross validation
grid_dt2 = GridSearchCV(estimator=dtentropy,  param_grid=params_dt,  scoring='roc_auc', cv=5, n_jobs=-1)
grid_dt2.fit(X_train2, y_train2)
best_hypers= grid_dt2.best_params_
print("Best hyperparams:\n", best_hypers)
#   {'criterion': 'gini', 'max_depth': 4}
best_CV_score = grid_dt2.best_score_
print("Best CV score:\n", best_CV_score)
#  0.7757795085621377
best_model = grid_dt2.best_estimator_
test_acc = best_model.score(X_test2, y_test2)
print("Test accuracy of best model dt: {:.3f}".format(test_acc))
# Best hyperparams:
#  {'criterion': 'entropy', 'max_depth': 8}
# Best CV score:
#  0.9910327013348018
# Test accuracy of best model dt: 0.975

###### Ada Boosting
# Ada estimates many times for the weak predicted labels.
ada = AdaBoostClassifier(base_estimator=dtentropy, n_estimators=180, random_state=1)
ada.fit(X_train2, y_train2)
y_pred_proba = ada.predict_proba(X_test2)[:,1]
# Evaluate test-set roc_auc_score
ada_roc_auc_dt_ent = roc_auc_score(y_test2, y_pred_proba)
print('ROC AUC score Decision Tree: {:.2f}'.format(ada_roc_auc_dt_ent))
# ROC AUC score Decision Tree: 1.00
# With labeling my target binary and AdaBoosting, my Decision Tree model has 1 accuracy. All labels correctly predicted.

#### Feature importance
importance = dtentropy.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
plt.bar([x for x in range(len(importance))], importance)
plt.show()
# Feature: 0, Score: 0.33853
# Feature: 1, Score: 0.23436
# Feature: 2, Score: 0.00000
# Feature: 3, Score: 0.14442
# Feature: 4, Score: 0.10124
# Feature: 5, Score: 0.18145
# Most important feature for DT are buying price, second maint price and safety.


#######################
#######################
#######################  Model 4 : Random Forrest  ###
#######################
#######################
rf = RandomForestClassifier(n_estimators=25, random_state=42)
rf.fit(X_train, y_train)   # multi class labels
predictionrf = rf.predict(X_test)
RFScore = accuracy_score(y_test, predictionrf)
print('Random Forrest Accuracy:' ,RFScore)
# Random Forrest Accuracy: 0.9595375722543352

rf2 = RandomForestClassifier(n_estimators=25, random_state=42)
rf2.fit(X_train2, y_train2)    # binary class labels
predictionrf2 = rf2.predict(X_test2)
RFScore2 = accuracy_score(y_test2, predictionrf2)
print('Random Forrest Accuracy' ,RFScore2)
# Random Forrest Accuracy: 0.9826589595375722

confusionrf = confusion_matrix(y_test2, predictionrf2)
print("Confusion Matrix\n")
print(confusionrf)
# Prediction looks fine.
print(classification_report(y_test2, predictionrf2))
# better predicted than decision tree.

###### Ada Boosting
adarf = AdaBoostClassifier(base_estimator=rf, n_estimators=180, random_state=1)
adarf.fit(X_train2, y_train2)
y_pred_proba_rf = adarf.predict_proba(X_test2)[:,1]
# Evaluate test-set roc_auc_score
ada_roc_auc_rf = roc_auc_score(y_test2, y_pred_proba_rf)
# Print roc_auc_score
print('ROC AUC score RandomForrest: {:.2f}'.format(ada_roc_auc_rf))
# ROC AUC score RandomForrest: 1.00

#### Feature importance
importances = pd.Series(data=rf.feature_importances_, index= X_train.columns)
importances_sorted = importances.sort_values()
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances Random Forrest')
plt.show()
# Most important feature for RF are safety, second persons and buying price. 
# Similar to the KNN but 2nd and 3th features are opposite. 


#######################
#######################
#######################  Model 5 : Gaussian Naive Bayes  ###
#######################
####################### 
# naive Bayes classifiers are a family of simple probabilistic classifiers based on applying 
# Bayes' theorem with strong (naive) independence assumptions between the features
gnb = GaussianNB()
gnb.fit(X_train,y_train)    # multi class label
predictionrgnb = gnb.predict(X_test)
GNBScore = accuracy_score(y_test, predictionrgnb)
print("Gaussian Naive Bayes Accuracy:",GNBScore)
# Gaussian Naive Bayes Accuracy: 0.605009633911368

confusiongnb = confusion_matrix(y_test, predictionrgnb)
print("Confusion Matrix\n")
print(confusiongnb)

classification_errors_gnb = (confusiongnb[0][1]  + confusiongnb[0][2]  + confusiongnb[0][3] + 
                              confusiongnb[1][0] + confusiongnb[1][2] + confusiongnb[1][3] + 
                              confusiongnb[2][0] + confusiongnb[2][1] + confusiongnb[2][3] + 
                              confusiongnb[3][0] + confusiongnb[3][1] + confusiongnb[3][2])
print(classification_errors_gnb)
# 205 missclassification out of 519. 40% labeled wrong.

gnb = GaussianNB()
gnb.fit(X_train2,y_train2)    # binary class label
predictionrgnb2 = gnb.predict(X_test2)
GNBScore2 = accuracy_score(y_test2, predictionrgnb2)
print("Gaussian Naive Bayes Accuracy:",GNBScore2)
# Gaussian Naive Bayes Accuracy: 0.928709055876686

confusiongnb2 = confusion_matrix(y_test2, predictionrgnb2)
print("Confusion Matrix\n")
print(confusiongnb2)
print(classification_report(y_test2, predictionrgnb2))

classification_errors_gnb2 = confusiongnb2[0][1]  + confusiongnb2[1][0]
print(classification_errors_gnb2)
# 37 missclassification out of 519. 7% labeled wrong. Label 1 isn't good.


# Instantiate lr
lr = LogisticRegression(C=0.01)
knn = KNeighborsClassifier(n_neighbors=8)
dt = DecisionTreeClassifier(max_depth=8, criterion="entropy", random_state = 100)
gnb = GaussianNB()
# Define the list classifiers
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt), ('Gaussian Naive Bayes', gnb)]
# Iterate over the pre-defined list of classifiers
for clf_name, clf in classifiers:    
    # Fit clf to the training set
    clf.fit(X_train2, y_train2)    
    # Predict y_pred
    y_pred = clf.predict(X_test2)
    # Calculate accuracy
    accuracy = accuracy_score(y_test2, y_pred)
    # Evaluate clf's accuracy on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy))
# Logistic Regression : 0.921
# K Nearest Neighbours : 0.948
# Classification Tree : 0.975
# Gaussian Naive Bayes : 0.929
# Classification Tree achieved the highest accuracy.

#########################################
###### ALL MODELS IN ONE TABLE ##########
#########################################

models = pd.DataFrame({
    'Model': ['KNN multi class', 'LogisticRegression multi class', 'DecisionTree multi class - gini', 
              'DecisionTree multi class - entropy', 'DecisionTree multi class - entropy- Ada', 'RandomForrest multi class',
              'NaiveBayes multi class', 'LogisticRegression binary class', 'DecisionTree binary class - entropy', 
              'RandomForrest binary class', 'RandomForrest binary class- Ada', 'NaiveBayes binary class'],
    'Score': [KNNTrainScore, lrscore, DTScoregini, 
              DTScoreentropy, ada_roc_auc_dt_ent, RFScore, 
              GNBScore, LRTrainScore, DTScoreentropy2,
              RFScore2, ada_roc_auc_rf, GNBScore2]})
models.sort_values(by='Score', ascending=False)

# end.