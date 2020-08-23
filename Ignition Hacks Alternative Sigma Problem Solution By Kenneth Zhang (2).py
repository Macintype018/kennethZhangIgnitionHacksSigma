#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 


# In[2]:


from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re


# In[3]:


trainingData = pd.read_csv('training_data.csv')
trainingData = trainingData[['Text','Sentiment']]


# In[4]:


trainingData = trainingData[trainingData.Sentiment != "Neutral"]
trainingData['Text'] = trainingData['Text'].apply(lambda x: x.lower())
trainingData['Text'] = trainingData['Text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

print(trainingData[ trainingData['Sentiment'] == 'Positive'].size)
print(trainingData[ trainingData['Sentiment'] == 'Negative'].size)

for idx,row in trainingData.iterrows():
    row[0] = row[0].replace('rt',' ')
    
max_features = 2000
tok = Tokenizer(num_words=max_features, split=' ')
tok.fit_on_texts(trainingData['Text'].values)
X = tok.texts_to_sequences(trainingData['Text'].values)
X = pad_sequences(X)


# In[5]:


embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


# In[6]:


Y = pd.get_dummies(trainingData['Sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[7]:


batch_size = 32
model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2)


# In[8]:


validation_size = 3000

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))


# In[9]:


pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X_validate)):
    
    result = model.predict(X_validate[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
   
    if np.argmax(result) == np.argmax(Y_validate[x]):
        if np.argmax(Y_validate[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1
       
    if np.argmax(Y_validate[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1

print("pos_acc", pos_correct/pos_cnt*100, "%")
print("neg_acc", neg_correct/neg_cnt*100, "%")


# In[ ]:


contestData = pd.read_csv("contestant_judgment.csv")

contestantText = contestData['Text']

contestantText = tok.texts_to_sequences(contestantText)

contestantText = pad_sequences(contestantText, maxlen=28, dtype='int32', value=0)
print(contestantText)

sentiment = model.predict(contestantText, batch_size=1,verbose = 2)[0]

if(np.argmax(sentiment) == 0):
    print("negative")
    
elif (np.argmax(sentiment) == 1):
    print("positive")


# In[ ]:




