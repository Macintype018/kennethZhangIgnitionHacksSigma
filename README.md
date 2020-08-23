# Ignition Hacks Sigma By: Kenneth Zhang

I submitted two different programs using different methods and algorithms to classify and predict the sentiment of sentences.
The first program submitted used traditional methods to classify and predict the sentiment of sentences whereas the second program submitted used a Sequential Model
to predict and classify sentiment values of the sentences. In this brief README file, I will go over the important segments of code and my thinking behind why I did it the way I did. If you have any further questions please email me at kzhang138@gmail.com. 


## Ignition Hacks Sigma Problem Solution 1 By: Kenneth Zhang

### Dataset Visualization and Pre-Processing
Firstly, I visualized the data provided to me. 
From the plot generated we can see that the distribution of sentiment values (0s and 1s) are even.

<pre><code>import pandas as pd
data = pd.read_csv('training_data.csv')
data.head()
data.info()
data.Sentiment.value_counts()

Sentiment_count=data.groupby('Sentiment').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['Text'])

plt.xlabel('Review Sentiments')
plt.ylabel('Number of Text Inputs')

plt.xticks([0,1])
plt.show()
</code></pre>

Next, I extracted the features from the text provided.
I converted the text into a matrix which visualizes the occurence of words in an individual text sample.
A matrix doc is created, and includes the number of times a word occurs in a given text sample. 

I then proceeded to split the data which is ideal for when I will assess the performance of the overall model. 
Generally, data is split into a training and test set.
Using Sklearn's most popular train_test_split function, we can easily pass three necessary parameters: features, target, and test_set size.
The count vectorizer converted the text in the text column into a vector of token counts. 

<pre><code>from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(data['Text'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    text_counts, data['Sentiment'], test_size=0.3, random_state=1)
    </code></pre>

### Building the Overall Model
Different algorithms in the Naive Baynes common group can be utilized in this case.
I tried many different algorithms such as the Bernoulli, Gaussian, and Complement Naive Baynes algorithm. 
Additionally, I attempted to use the RandomForestClassifer to classify and predict the sentiments for the sentences, but proved to achieve a lower accuracy.
In the end, I was deciding between the Gaussian, Complement, and Multinomial Naive Baynes Algorithms.

<pre><code>from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics

clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))
print(predicted)

clf.intercept_
clf.predict(X_train)
</code></pre>

In my solution, I used the Multinomial Naive Baynes algorithm, but it can be interchangeable with the Gaussian, Bernoulli, and Complement Algorithms.
Although they are all Naive Baynes algorithms, they do have differences and are optimized for different scenarios.

The Multinomial Naive Baynes Classifier assumes that features of the text are drawn from Multinomial Distribution.
Scikit Learn has made it very easy to use the functions to implement the Multinomial Naive Baynes Classifier and can be accessed simply by using sklearn.naive_baynes.GaussianNB.
The MultinomialNB module is first imported then used to fit the X_train and y_train values. 
But, I must fit the model first before implementing the .predict method. 



### Generating our Features
I proceeded to split the data for the text classification model, again, splitting the data into the same three parameters mentioned in previous sections.
I will use the TFidVectorizer to extract those features from the text samples.

<pre><code>from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
text_tf = tf.fit_transform(data['Text'])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    text_tf, data['Sentiment'], test_size=0.3, random_state=123)
</code></pre>

I will continue to use the MultinomialNB but other Naive Baynes algorithms are interchangeable depending on the dataset provided.
The program was executed and produced an accuracy of approx. 81.2% at the time of testing.

<pre><code>from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, predicted))
</code></pre>

Alternatively, we can use the Random Forest Classifier, but it did not achieve an accuracy as high as the Multinomial or Bernoulli NB Classifiers.

<pre><code>from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(clf.predict(X_test))

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
</code></pre>


## Ignition Hacks Sigma Problem Solution 2 By: Kenneth Zhang

### Building a Recurrent Neural Network for Text Classification 
After importing the necessary modules to pre-process and construct the model, I began vectorizing and tokenizing the text. 
To improve the accuracy of the overall RNN, I attempted to fiter the data of any 'neutral' text, meaning text that did not show negativity or positivity.
This resulted in only valid text and words remained in the dataset therefore improving the overall accuracy.

<pre><code>trainingData = pd.read_csv('training_data.csv')
trainingData = trainingData[['Text','Sentiment']]
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
</code></pre>

Next, I constructed the Sequential Model.
The Sequential Model consisted of Embedding, Spatial Dropout, Dense, and Long Short Term Memory layer(s).

<pre><code>
model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())
</code></pre>

I trained the model with 7 epochs, but I wanted to use 10.
<pre><code>model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2)
</code></pre>

After extracting a validation set, we can measure the overall score and accuracy of the model.
<pre><code>
validation_size = 1500
X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))
</code></pre>

The Neural Network had a significantly higher accuracy than the traditional Naive Baynes classifiers.















