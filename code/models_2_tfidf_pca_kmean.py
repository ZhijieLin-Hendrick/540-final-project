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


##########################################################
# in this part, we were trying to find the proper parameter for PCA
##########################################################

tv_1 = TfidfVectorizer()
tv_1_fit = tv_1.fit_transform(X_train)
X_train_tfidf = tv_1_fit.toarray()

expl = []
for i in tqdm(range(100,2100,400)):
  pca = PCA(n_components=i)
  X_train_reduced = pca.fit_transform(X_train_tfidf)
  expl.append(sum(pca.explained_variance_ratio_))

pca = PCA(n_components=1600)
X_train_reduced = pca.fit_transform(X_train_tfidf)
sum(pca.explained_variance_ratio_)

##########################################################
# in this part, we were trying to find the proper parameter for KMeans
##########################################################

sse = []
for k in tqdm(range(1, 26)):
  estimator = KMeans(n_clusters=k)
  estimator.fit(np.array(X_reduced))
  sse.append(estimator.inertia_)
X = range(1, 26)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X, sse, 'o-')
plt.show()


############################################################
# This part is the model
# We played with some parameters of tfidf, but we finally decided to just use the default (which is norm='l2', max_feature=None). So, for the output of this function, we just collected the last one row data.
############################################################
def train_models_kmeans(X_train, X_test, y_train, y_test, n):
    model_names = ['DecisionTree', 'SVC', 'LogisticRegression', 'SGDClassifier']
    # decision_tree
    model1 = tree.DecisionTreeClassifier()
    # naive_bayes
    # model2 = naive_bayes.MultinomialNB()
    # SVC
    model3 = SVC(kernel='linear')
    # LogisticRegression
    model4 = linear_model.LogisticRegression(solver='lbfgs')
    # linear_SGD_classifier
    model5 = linear_model.SGDClassifier()

    models = [model1, model3, model4, model5]

    norm = ['l2']
    max_feature = [None]
    params = [(i, j) for i in norm for j in max_feature]

    scores = pd.DataFrame(index=params, columns=model_names)
    f1_df = pd.DataFrame(index=params, columns=model_names)
    for k in range(len(models)):
      for j in range(len(params)):
        tv1 = TfidfVectorizer(norm=params[j][0], max_features=params[j][1])
        tv1_fit = tv1.fit_transform(X_train)
        X1_train = tv1_fit.toarray()

        pca = PCA(n_components=1600)
        X1_train_reduced = pca.fit_transform(X1_train)

        estimator = KMeans(n)
        estimator.fit(np.array(X1_train_reduced))
        X_train_kmeans = pd.DataFrame(X1_train_reduced)
        X_train_kmeans['kmeans'] = estimator.labels_

        X_train_kmeans_group = {}
        X_train_kmeans_group_label = {}
        groups = X_train_kmeans['kmeans'].unique()
        for i in groups:
          indx = list(X_train_kmeans['kmeans']==i)
          X_train_kmeans_group[i]=X_train_kmeans[indx]
          X_train_kmeans_group_label[i] = y_train[indx]

        model_group = {}
        for i in groups:
          if X_train_kmeans_group_label[i].nunique()==2:
            model_group[i] = models[k].fit(X_train_kmeans_group[i], X_train_kmeans_group_label[i])
          elif X_train_kmeans_group_label[i].nunique()==1:
            model_group[i] = list(X_train_kmeans_group_label[i])[0]
          else:
            print('ERROR\n')

        X1_test = tv1.transform(X_test).toarray()

        X1_test_reduced = pca.transform(X1_test)

        test_label = estimator.predict(X1_test_reduced)
        X_test_kmeans = pd.DataFrame(X1_test_reduced)
        X_test_kmeans['kmeans'] = test_label

        y_predict = {}
        y_label = {}
        acc = {}
        f1_score = {}
        for i in groups:
          if not isinstance(model_group[i], int):
            indx = list(X_test_kmeans['kmeans']==i)
            if sum(indx)!=0:
              y_predict[i] = model_group[i].predict(X_test_kmeans[indx])
              acc[(i,sum(indx))] = sklearn.metrics.accuracy_score(y_test[indx], y_predict[i])
              f1_score[(i,sum(indx))] = sklearn.metrics.f1_score(y_test[indx], y_predict[i])
            else:
              acc[(i,0)] = 0
              f1_score[(i,0)] = 0
          else:
            indx = list(X_test_kmeans['kmeans']==i)
            if sum(indx)!=0:
              acc[(i,sum(indx))] = sum(y_test[indx]==model_group[i])/len(indx)
              f1_score[(i,sum(indx))] = 2*acc[(i,sum(indx))]/(acc[(i,sum(indx))]+1)
            else:
              acc[(i,0)] = 0
              f1_score[(i,0)] = 0

        total = 0
        for i in acc.keys():
          total+=i[1]
        score = 0
        f1 = 0
        for key, value in acc.items():
          score+=value*(key[1]/total)
        scores.iloc[j,k] = score
        for key, value in f1_score.items():  
          f1+=value*(key[1]/total)
        f1_df.iloc[j,k] = f1

    return scores, f1_df


# scores for PCA(n_components=1600) + kmeans(k=2)
acc_2means, f1_2means= train_models_kmeans(X_train, X_test, y_train, y_test, 2)

# scores for PCA(n_components=1600) + kmeans(k=5)
acc_5means, f1_5means= train_models_kmeans(X_train, X_test, y_train, y_test, 5)