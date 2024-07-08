import numpy as np
import pandas as pd

df= pd.read_csv("/Users/niramaypatel/Desktop/spam.csv", encoding = 'latin-1')
df.rename(columns={'v1':'labels','v2':'message'},inplace='True')
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
df['labels']=df['labels'].map({'ham':0,'spam':1})
df['wordcount']=df['message'].apply(lambda x: len(x.split()))

def currency_symbol(data):
    currency = ['$','€','£','₹']
    for i in currency:
        if i in data:
            return 1
    return 0
df["currency_in_df"]=df["message"].apply(currency_symbol)

only_spams=df[df['labels']==1]
only_hams=df[df['labels']==0]
count= int(len(only_hams)/len(only_spams))
for i in range(0,count-1):
    df=pd.concat([df,only_spams])
def number(data):
    for i in data:
        if ord(i)>=48 and ord(i)<=57:
            return 1
    return 0

df["contains_number"]=df['message'].apply(number)

#data cleaning
import nltk
import re
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
corpus=[]
wnl=WordNetLemmatizer()
for sms in list(df.message):
    message=re.sub(pattern='[^a-zA-Z]',repl=' ',string=sms)
    message=message.lower()
    words=message.split()
    fw=[word for word in words if word not in set(stopwords.words('english'))]
    lmw=[wnl.lemmatize(word) for word in fw]
    message=' '.join(lmw)
    corpus.append(message)

# assigning test and train data to model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(max_features=500)
vectors=tfidf.fit_transform(corpus).toarray()
feature_name=tfidf.get_feature_names_out()
X=pd.DataFrame(vectors,columns=feature_name)
y=df['labels']
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import classification_report,confusion_matrix
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# Naive Bayes Model
from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()
cv=cross_val_score(mnb,X,y,scoring='f1',cv=10)
mnb.fit(X_train,y_train)
y_pred=mnb.predict(X_test)
cm=confusion_matrix(y_test,y_pred)

""" Uncomment the below text to see the plot"""
# import matplotlib.pyplot as plt
# import seaborn as sns
# %matplotlib inline
# plt.figure(figsize=(8,8))
# axis_labels=['ham','spam']
# g=sns.heatmap(data=cm,xticklabels=axis_labels,yticklabels=axis_labels,annot=True, fmt='g', cmap='Blues')
# p=plt.title("Confusion matrix of Multinomial Naive Bayes")
# p=plt.xlabel("Actual Values")
# p=plt.ylabel("Predicted values")
# plt.show()

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
cv1=cross_val_score(dt,X,y,scoring='f1',cv=10)
dt.fit(X_train.values,y_train.values)
y_pred1=dt.predict(X_test)
cm=confusion_matrix(y_test,y_pred1)

""" Uncomment the below text to see the plot"""
# import matplotlib.pyplot as plt
# import seaborn as sns
# %matplotlib inline
# plt.figure(figsize=(8,8))
# axis_labels=['ham','spam']
# g=sns.heatmap(data=cm,xticklabels=axis_labels,yticklabels=axis_labels,annot=True, fmt='g', cmap='Blues')
# p=plt.title("Confusion matrix of Multinomial Naive Bayes")
# p=plt.xlabel("Actual Values")
# p=plt.ylabel("Predicted values")
# plt.show()

def predict_spam(sms):
    message=re.sub(pattern='[^a-zA-Z]',repl=' ',string=sms)
    message=message.lower()
    words=message.split()
    fw=[word for word in words if word not in set(stopwords.words('english'))]
    lmw=[wnl.lemmatize(word) for word in fw]
    message=' '.join(lmw)
    temp =tfidf.transform([message]).toarray()
    return dt.predict(temp)

# predication
sample_message= input("Enter the sms: ")
if predict_spam(sample_message):
    print("This is a spam message")
else:
    print("This is ham (normal) message")
