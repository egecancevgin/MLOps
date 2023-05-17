import pandas as pd
import numpy as np
import re
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

warnings.simplefilter("ignore")
data = pd.read_csv("Language Detection.csv")

X = data["Text"]
y = data["Language"]

le = LabelEncoder()
y = le.fit_transform(y)

data_list = []
for text in X:
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    text = text.lower()
    data_list.append(text)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

cv = CountVectorizer()
cv.fit(X_train)

x_train = cv.transform(X_train).toarray()
x_test = cv.transform(X_test).toarray()

# First way of representation and evaluation
model = MultinomialNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

# Second way of representation and evaluation
pipe = Pipeline([('vectorizer', cv), ('multinomialNB', model)])
pipe.fit(X_train, y_train)

y_pred2 = pipe.predict(X_test)
ac2 = accuracy_score(y_test, y_pred2)

with open('trained_pipeline-0.1.0.pkl', 'wb') as f:
    pickle.dump(pipe, f)

# !zip -r ./trained_pipeline-0.1.0.pkl.zip ./trained_pipeline-0.1.0.pkl

# Testing
text = "Hello, how are you?"
text = "Ciao, come stai?"

y = pipe.predict([text])
le.classes_[y[0]], y