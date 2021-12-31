import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm,naive_bayes, neighbors
from sklearn.model_selection  import train_test_split
from stemming.porter2 import stem

df=pd.read_csv("IRminiProj#2_dataset90.csv",encoding="ISO-8859-1")
cv1=TfidfVectorizer(stop_words="english", lowercase=True)
x=df["Abstract"]
y=df["Y"]
x = [" ".join([stem(word) for word in sentence.split(" ")]) for sentence in x]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=4)
x_train=cv1.fit_transform(x_train)
x_train=x_train.toarray()

print(x_train)

# clf=naive_bayes.MultinomialNB(alpha=0.5)
# clf.fit(x_train,y_train)
# x_testcv=cv1.transform(x_test)
# x_testcv=x_testcv.toarray()
# pred=clf.predict(x_testcv)
# print(pred)
# print(np.array(y_test))
# print(clf.score(x_testcv,y_test))
#
# b_nb=naive_bayes.BernoulliNB()
# b_nb.fit(x_train,y_train)
# x_testcv=cv1.transform(x_test)
# x_testcv=x_testcv.toarray()
# pred=b_nb.predict(x_testcv)
# print(pred)
# print(np.array(y_test))
# print(b_nb.score(x_testcv,y_test))

# l_svm=svm.SVC(kernel="linear")
# l_svm.fit(x_train,y_train)
# x_testcv=cv1.transform(x_test)
# x_testcv=x_testcv.toarray()
# pred=l_svm.predict(x_testcv)
# print(pred)
# print(np.array(y_test))
# print(l_svm.score(x_testcv,y_test))

# l_svm=svm.SVC(kernel="rbf")
# l_svm.fit(x_train,y_train)
# x_testcv=cv1.transform(x_test)
# x_testcv=x_testcv.toarray()
# pred=l_svm.predict(x_testcv)
# print(pred)
# print(np.array(y_test))
# print(l_svm.score(x_testcv,y_test))

knn=neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
x_testcv=cv1.transform(x_test)
x_testcv=x_testcv.toarray()
pred=knn.predict(x_testcv)
print(pred)
print(np.array(y_test))
print(knn.score(x_testcv,y_test))
