#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np 
import pandas as pd 
import re
import nltk 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

trainingData = pd.read_csv('training_data.csv')


# In[30]:


trainingData.head()


# In[31]:


sentiment = trainingData['Sentiment']
sentiment


# In[32]:


nltk.__version__


# In[33]:


import pandas as pd
data = pd.read_csv('training_data.csv')
data.head()
data.info()
data.Sentiment.value_counts()


# In[34]:


Sentiment_count=data.groupby('Sentiment').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['Text'])

plt.xlabel('Review Sentiments')
plt.ylabel('Number of Text Inputs')

plt.xticks([0,1])
plt.show()


# In[35]:


from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(data['Text'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    text_counts, data['Sentiment'], test_size=0.3, random_state=1)


# In[36]:


from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics

clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))
print(predicted)

clf.intercept_
clf.predict(X_train)


# In[37]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
text_tf = tf.fit_transform(data['Text'])


# In[38]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    text_tf, data['Sentiment'], test_size=0.3, random_state=123)


# In[39]:


from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, predicted))


# In[40]:


from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, predicted))


# In[41]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(clf.predict(X_test))

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[42]:


plt.scatter(y_test, predicted)


# In[43]:


contestData = pd.read_csv('contestant_judgment.csv')
print(contestData)

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(contestData['Text'])

from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
text_tf = tf.fit_transform(data['Text'])

from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics

clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))
print(predicted)


# In[45]:


import pandas as pd
list1 = {'Sentiments':predicted}

df = pd.DataFrame(list1)
df_csv = pd.read_csv('contestant_judgment.csv')
df_csv['Sentiments'] = df.Sentiments 
df_csv.to_csv('contestant_judgment.csv', index=False, mode= 'w')


# In[ ]:




