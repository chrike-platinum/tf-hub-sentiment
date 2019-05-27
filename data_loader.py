import os
import pandas as pd
import numpy as np
import time
#from nltk.corpus import stopwords

dataframe = pd.read_csv("/Users/christiaanleysen/Downloads/trainingandtestdata/training.1600000.processed.noemoticon.csv", encoding = "ISO-8859-1", header=None).iloc[:, [0, 4, 5]].sample(frac=1).reset_index(drop=True)
#print(dataframe.info())
#print(dataframe.head())


import re
def preprocess_tweet(tweet):
    #Preprocess the text in a single tweet
    #arguments: tweet = a single tweet in form of string
    #convert the tweet to lower case
    tweet.lower()
    #convert all urls to sting "URL"
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #convert all @username to "AT_USER"
    tweet = re.sub('@[^\s]+','', tweet)
    #correct all multiple white spaces to a single white space
    tweet = re.sub('[\s]+', ' ', tweet)
    #convert "#topic" to just "topic"
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = re.sub(r'\W*\b\w{1,3}\b', '', tweet)
    tweet = re.sub(r'&quot','', tweet)
    return tweet.strip()

def transform_sentiment(sentiment):
    if sentiment=='negative':
        return 0
    if sentiment=='positive':
        return 1



test_data=pd.read_csv('/Users/christiaanleysen/Downloads/twitter-airline-sentiment/Tweets.csv')
test_data=test_data[['text','airline_sentiment','airline_sentiment_confidence']]

test_data=test_data[test_data['airline_sentiment_confidence']==1]
print(len(test_data))
print(test_data['airline_sentiment'].value_counts())
test_data = test_data.groupby('airline_sentiment')
test_data=test_data.apply(lambda x: x.sample(test_data.size().min())).reset_index(drop=True)
print(test_data['airline_sentiment'].value_counts())


df2=test_data[test_data['airline_sentiment'].isin(['positive','negative'])]
df2=df2[['airline_sentiment','text']]
df2.columns=['sentiment','tweet']
df2.tweet=df2.tweet.apply(preprocess_tweet)
df2.sentiment=df2.sentiment.apply(transform_sentiment)
print(df2)



now = time.time()

'''
#users = np.array(dataframe.iloc[:, 1].values)
tweets = np.array(dataframe.iloc[:, 2].apply(preprocess_tweet).values)
print(len(tweets))
sentiment = np.array(dataframe.iloc[:, 0].values)
print(len(sentiment))
sentiment[sentiment>3]=1

data_set=pd.DataFrame(np.column_stack([sentiment, tweets]),columns=['sentiment', 'tweet'])
'''

df1=pd.read_csv("/Users/christiaanleysen/Downloads/trainingandtestdata/train.csv", encoding='latin1')
df1 = df1.drop('ItemID', axis=1)
df1 = df1.sample(frac=1)
df1.columns = ['sentiment', 'tweet']
df1.tweet=df1.tweet.apply(preprocess_tweet)
print(df1)





#df_total = pd.concat([data_set,df1,df2])
df_total = pd.concat([df1,df2])
df_total = df_total[df_total['tweet']!= '']
df_total = df_total.sample(frac=1).reset_index(drop=True)

g = df_total.groupby('sentiment')
df_total=g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
df_total = df_total.sample(frac=1).reset_index(drop=True)

print('balance:',df_total.sentiment.value_counts())
df_total.to_csv('twitter_data_small.csv',index=False)
#filtered_words = [word for word in word_list if word not in stopwords.words('english')]

print('Duration',time.time()-now)

