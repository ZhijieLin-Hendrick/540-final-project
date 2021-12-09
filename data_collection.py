import os
import csv
import pandas as pd
from pandas.core.frame import DataFrame
l=[]
K=[]

for i in os.listdir(r'/Users/chaixiaotang/Documents/GitHub/540-final-project5/fake'):
    if i[0:5]=='polit':
        l.append(i)
    for j in range(len(l)):
        folder = r'/Users/chaixiaotang/Documents/GitHub/540-final-project5/fake/'+l[j]+'/retweets'
        if os.path.isdir(folder):
            for k in os.listdir(folder):
                if k[0].isdigit():
                    K.append(k)
tweet_id=[]
for w in K:
    t_id=w.split('.')
    tweet_id.append(t_id)
df=DataFrame(tweet_id,columns=['tweet_id','json'])
df2=df['tweet_id']
df2.to_csv("total_tweet_id_final",index=False)
dfnew=df2.sample(2000)
dfnew.to_csv("total_tweet_id_final1.csv",index=False)
