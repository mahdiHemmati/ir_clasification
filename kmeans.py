import timeit

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection  import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("**** K-means clustering based on Y label ****")
start = timeit.default_timer()

df=pd.read_csv("IRminiProj#2_dataset.csv",encoding="ISO-8859-1")
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, max_features=5000)

X = df["Abstract"]
Y = df["Y"]

after_preprocess = timeit.default_timer()
print('Load and Preprocess Time: ', after_preprocess - start)

domain = df["Domain"]
filtered_x = []
filtered_y = []

for idx, val in enumerate(domain):
    if val.strip() == "CS":
        filtered_x.append(X[idx])
        filtered_y.append(Y[idx])


x_train,x_test, y_train, y_test = train_test_split(filtered_x, filtered_y, test_size=0.3,random_state=4)

x_train = vectorizer.fit_transform(x_train)

Y = vectorizer.transform(x_test)
Y = Y.toarray()
y_test_array = np.array(y_test)

for i in [7, 10, 14, 17, 19]:
    print("**** K: %d ****" % i)

    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    km.fit(x_train)

    after_train = timeit.default_timer()
    print('Train Time: ', after_train - start)

    pred = km.predict(Y)

    after_test = timeit.default_timer()
    print('Test Time: ', after_test - start)

    score = silhouette_score(x_train, km.labels_, metric='euclidean')
    print('Precision (macro-average) is: %.3f' % precision_score(y_test_array, pred, average='macro'))
    print('Precision (micro-average) is: %.3f' % precision_score(y_test_array, pred, average='micro'))
    print('Recall (macro-average) is: %.3f' % recall_score(y_test_array, pred, average='macro'))
    print('Recall (micro-average) is: %.3f' % recall_score(y_test_array, pred, average='micro'))
    print('Accuracy is: %.3f' % accuracy_score(y_test_array, pred))