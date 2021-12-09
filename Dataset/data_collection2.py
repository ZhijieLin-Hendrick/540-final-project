import tweepy
import json
import pandas as pd
from pandas.core.frame import DataFrame
API_Key = 'OaqhRT7qeFHt4Ye3lrgS8hAog'
API_Key_Secret = 'Wvzpewg3TVMWcexYhz6OgxalLQCobTrVeJ4gnbmnk4GiYj0IZG'
Access_Token = '1452851797912010759-0QOY2qp5HpbINHeQJS2rUz4wrUTo6K'
Access_Secret = 'KE0PqlIivQdFMFE25CQsGKF4KwXEdH4jLrkC2s8N34BdK'

authentication = tweepy.OAuthHandler(API_Key, API_Key_Secret)
authentication.set_access_token(Access_Token, Access_Secret)
api = tweepy.API(authentication)
df1=pd.read_csv("/Users/chaixiaotang/Desktop/FakeNewsNet-master/dataset/total_tweet_id_final1.csv")
df1.head()
full_text= []
in_reply_to_status_id=[]
in_reply_to_user_id = []
in_reply_to_screen_name = []
author_id = []
author_name = []
author_screen_name = []
author_followers_count = []
author_friends_count = []
account_created_time = []
author_favourites_count = []
author_statuses_count = []
favorite_count = []
for x in range(2000):
    id1 = df1['tweet_id'][x]
    append_flag = True
    try:
        status = api.get_status(id1, tweet_mode="extended")
        # L.append(status.full_text)
    except BaseException as e:
        append_flag = False
        print('except:', e)
        full_text.append(None)
        in_reply_to_status_id.append(None)
        in_reply_to_user_id.append(None)
        in_reply_to_screen_name.append(None)
        author_id.append(None)
        author_name.append(None)
        author_screen_name.append(None)
        author_followers_count.append(None)
        author_friends_count.append(None)
        account_created_time.append(None)
        author_favourites_count.append(None)
        author_statuses_count.append(None)
        favorite_count.append(None)
    if append_flag:
        full_text.append(status.full_text)
        in_reply_to_status_id.append(status.in_reply_to_status_id)
        in_reply_to_user_id.append(status.in_reply_to_user_id)
        in_reply_to_screen_name.append(status.in_reply_to_screen_name)
        author_id.append(status.author.id)
        author_name.append(status.author.name)
        author_screen_name.append(status.author.screen_name)
        author_followers_count.append(status.author.followers_count)
        author_friends_count.append(status.author.friends_count)
        account_created_time.append(status.author.created_at)
        author_favourites_count.append(status.author.favourites_count)
        author_statuses_count.append(status.author.statuses_count)
        favorite_count.append(status.favorite_count)
c={"full_text" : full_text,
   "in_reply_to_status_id": in_reply_to_status_id,
   "in_reply_to_user_id":in_reply_to_user_id,
   "in_reply_to_screen_name":in_reply_to_screen_name,
   "author_id":author_id,
   "author_name":author_name,
   "author_screen_name":author_screen_name,
   "author_followers_count":author_followers_count,
   "author_friends_count":author_friends_count,
   "account_created_time":account_created_time,
   "author_favourites_count":author_favourites_count,
   "author_statuses_count":author_statuses_count,
   "favorite_count":favorite_count}
d=DataFrame(c)
dftext=pd.concat([df1, d], axis=1)
dftext.to_csv("politi_faketext.csv", encoding='utf_8_sig',index=False)
