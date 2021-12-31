import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection  import train_test_split
from stemming.porter2 import stem
import timeit
import warnings
warnings.filterwarnings('ignore')

print("**** Linear Svm classification based on Y label ****")
start = timeit.default_timer()

df = pd.read_csv("IRminiProj#2_dataset.csv",encoding="ISO-8859-1")
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
x = df["Abstract"]
y = df["Y"]
y1 = df["Y1"]
y2 = df["Y2"]
y12 = []

for idx, val in enumerate(y1):
    y12.append(str(val) + str(y2[idx]))

x = [" ".join([stem(word) for word in sentence.split(" ")]) for sentence in x]

after_preprocess = timeit.default_timer()
print('Load and Preprocess Time: ', after_preprocess - start)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=4)
x_train = vectorizer.fit_transform(x_train)
x_train = x_train.toarray()

l_svm = svm.SVC(kernel="linear")
l_svm.fit(x_train, y_train)

after_train = timeit.default_timer()
print('Train Time: ', after_train - start)

x_testcv = vectorizer.transform(x_test)
x_testcv = x_testcv.toarray()
pred=l_svm.predict(x_testcv)

after_test = timeit.default_timer()
print('Test Time: ', after_test - start)

y_test_array = np.array(y_test)

print('Precision (macro-average) is: %.3f' % precision_score(y_test_array, pred, average='macro'))
print('Precision (micro-average) is: %.3f' % precision_score(y_test_array, pred, average='micro'))
print('Recall (macro-average) is: %.3f' % recall_score(y_test_array, pred, average='macro'))
print('Recall (micro-average) is: %.3f' % recall_score(y_test_array, pred, average='micro'))
print('Accuracy is: %.3f' % accuracy_score(y_test_array, pred))


print("**** Gaussian Svm classification based on Y label ****")
start = timeit.default_timer()

g_svm = svm.SVC(kernel="rbf")
g_svm.fit(x_train, y_train)

after_train = timeit.default_timer()
print('Train Time: ', after_train - start)

x_testcv = vectorizer.transform(x_test)
x_testcv = x_testcv.toarray()
pred = g_svm.predict(x_testcv)

after_test = timeit.default_timer()
print('Test Time: ', after_test - start)

y_test_array = np.array(y_test)

print('Precision (macro-average) is: %.3f' % precision_score(y_test_array, pred, average='macro'))
print('Precision (micro-average) is: %.3f' % precision_score(y_test_array, pred, average='micro'))
print('Recall (macro-average) is: %.3f' % recall_score(y_test_array, pred, average='macro'))
print('Recall (micro-average) is: %.3f' % recall_score(y_test_array, pred, average='micro'))
print('Accuracy is: %.3f' % accuracy_score(y_test_array, pred))

# ********************************************************************* #

print("**** Svm classification based on Y1 label ****")
start = timeit.default_timer()

x_train,x_test,y_train,y_test = train_test_split(x,y1,test_size=0.3,random_state=4)
x_train = vectorizer.fit_transform(x_train)
x_train = x_train.toarray()

l_svm = svm.SVC(kernel="linear")
l_svm.fit(x_train, y_train)

after_train = timeit.default_timer()
print('Train Time: ', after_train - start)

x_testcv = vectorizer.transform(x_test)
x_testcv = x_testcv.toarray()
pred=l_svm.predict(x_testcv)

after_test = timeit.default_timer()
print('Test Time: ', after_test - start)

y_test_array = np.array(y_test)

print('Precision (macro-average) is: %.3f' % precision_score(y_test_array, pred, average='macro'))
print('Precision (micro-average) is: %.3f' % precision_score(y_test_array, pred, average='micro'))
print('Recall (macro-average) is: %.3f' % recall_score(y_test_array, pred, average='macro'))
print('Recall (micro-average) is: %.3f' % recall_score(y_test_array, pred, average='micro'))
print('Accuracy is: %.3f' % accuracy_score(y_test_array, pred))

# ********************************************************************* #

print("**** Svm classification based on Y1 and Y2 label ****")
start = timeit.default_timer()

x_train,x_test,y_train,y_test = train_test_split(x,y12,test_size=0.3,random_state=4)
x_train = vectorizer.fit_transform(x_train)
x_train = x_train.toarray()

l_svm = svm.SVC(kernel="linear")
l_svm.fit(x_train, y_train)

after_train = timeit.default_timer()
print('Train Time: ', after_train - start)

x_testcv = vectorizer.transform(x_test)
x_testcv = x_testcv.toarray()
pred=l_svm.predict(x_testcv)

after_test = timeit.default_timer()
print('Test Time: ', after_test - start)

y_test_array = np.array(y_test)

print('Precision (macro-average) is: %.3f' % precision_score(y_test_array, pred, average='macro'))
print('Precision (micro-average) is: %.3f' % precision_score(y_test_array, pred, average='micro'))
print('Recall (macro-average) is: %.3f' % recall_score(y_test_array, pred, average='macro'))
print('Recall (micro-average) is: %.3f' % recall_score(y_test_array, pred, average='micro'))
print('Accuracy is: %.3f' % accuracy_score(y_test_array, pred))

