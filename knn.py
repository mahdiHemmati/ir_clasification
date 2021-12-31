import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import neighbors
from sklearn.model_selection  import train_test_split
from stemming.porter2 import stem

df = pd.read_csv("IRminiProj#2_dataset.csv",encoding="ISO-8859-1")
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
x = df["Abstract"]
y = df["Y"]

x = [" ".join([stem(word) for word in sentence.split(" ")]) for sentence in x]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=4)
x_train = vectorizer.fit_transform(x_train)
x_train = x_train.toarray()

x_testcv = vectorizer.transform(x_test)
x_testcv = x_testcv.toarray()
y_test_array = np.array(y_test)

for i in [3, 5, 10]:
    knn = neighbors.KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred = knn.predict(x_testcv)
    print('Precision (macro-average) is: %.3f' % precision_score(y_test_array, pred, average='macro'))
    print('Precision (micro-average) is: %.3f' % precision_score(y_test_array, pred, average='micro'))
    print('Recall (macro-average) is: %.3f' % recall_score(y_test_array, pred, average='macro'))
    print('Recall (micro-average) is: %.3f' % recall_score(y_test_array, pred, average='micro'))
    print('Accuracy is: %.3f' % accuracy_score(y_test_array, pred))
