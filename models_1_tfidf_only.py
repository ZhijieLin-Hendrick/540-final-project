import os
import datetime
import collections
import numpy as np
from tqdm import tqdm
import sklearn
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree
from sklearn import naive_bayes
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA


file111 = 'train.csv'
file222 = 'test.csv'
file333 = 'val.csv'

train = pd.read_csv(file111)
test = pd.read_csv(file222)
val = pd.read_csv(file333)

train_new = pd.concat([train,val])

train_new = train_new.dropna().reset_index(drop=True)
train_new = train_new.drop('Unnamed: 0',axis=1)
test_kmeans111 = test_kmeans111.dropna().reset_index(drop=True)
test_kmeans111 = test_kmeans111.drop('Unnamed: 0',axis=1)

X_train = train_new.full_text
X_test = test_kmeans111.full_text
y_train = train_new.label
y_test = test_kmeans111.label


# We played with some parameters of tfidf, but we finally decided to just use the default (which is norm='l2', max_feature=None). So, for the output of this function, we just collected the last one row data.
def train_models_new(X_train, X_test, y_train, y_test):
    model_names = ['DecisionTree', 'SVC', 'LogisticRegression', 'SGDClassifier']
    # decision_tree
    model1 = tree.DecisionTreeClassifier()
    # naive_bayes
    #model2 = naive_bayes.MultinomialNB()
    # SVC
    model3 = SVC(kernel='linear')
    # LogisticRegression
    model4 = linear_model.LogisticRegression(solver='lbfgs')
    # linear_SGD_classifier
    model5 = linear_model.SGDClassifier()

    models = [model1, model3, model4, model5]

    norm = ['l1', 'l2']
    max_feature = [500, 1000, 2000, None]
    params = [(i, j) for i in norm for j in max_feature]

    acc = pd.DataFrame(index=params, columns=model_names)
    f1 = pd.DataFrame(index=params, columns=model_names)
    for i in range(len(models)):
      for j in range(len(params)):
        tv1 = TfidfVectorizer(norm=params[j][0], max_features=params[j][1])
        tv1_fit = tv1.fit_transform(X_train)
        X1_train = tv1_fit.toarray()
        model1 = models[i].fit(X1_train, y_train)
        X1_test = tv1.transform(X_test).toarray()
        y_predict = model1.predict(X1_test)

        acc.iloc[j,i] = sklearn.metrics.accuracy_score(y_test, y_predict)
        f1.iloc[j,i] = sklearn.metrics.f1_score(y_test, y_predict)

    return acc,f1


# scores for td-idf only model
acc_1,f1_1 = train_models_new(X_train, X_test, y_train, y_test)

# scores for tf-idf model with shuffled label
random.shuffle(y_train)
acc_2,f1_2 = train_models_new(X_train, X_test, y_train, y_test)