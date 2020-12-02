import pandas as pd

messages = pd.read_csv('SMSSpamCollection.txt', sep='\t',
                           names=["label", "message"])


import nltk
from nltk.corpus import stopwords

import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
corpus = []
wn = WordNetLemmatizer()
tf = TfidfVectorizer(max_features = 3000)

stopwords = stopwords.words('english')
for i in range (0,len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [wn.lemmatize(modda) for modda in review if not modda in stopwords]
    review = ' '.join(review)
    corpus.append(review)
    
x = tf.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)


model.fit(x_train,y_train)

y_pred = model.predict(x_test)
score = model.score(y_test,y_pred)





 
    


