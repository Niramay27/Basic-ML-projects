# numpy,pandas,nltk with stopwords, sckitlearn should be installed in your pc to run the following code...

import string
import numpy as np
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
stopwords = nltk.corpus.stopwords.words('english')
stemmer = nltk.SnowballStemmer("english")
dataset = pd.read_csv("/Users/niramaypatel/Desktop/labeled_data.csv")

# there is no null row or column here....checked
# stopwords like and,the,is...etc are removed with importing stopwords
# all verbs are converted to their base form using stemmer

dataset["labels"] = dataset["class"].map({0: "Hate Speech",
                                          1: "Offensive Language",
                                          2: "Neither hate nor offensive"})
data = dataset[['tweet', 'labels']]


# data cleaning by removing unnecessary words
def data_cleaning(text):
    text = str((text).lower())                       # converting upper to lower as python is case sensitive
    text = re.sub(r'https?://[www\.]*', "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = [word for word in text.split(" ") if word not in stopwords]      # remove stopwords
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(" ")]                 # remove stemmers
    text = " ".join(text)
    return text


data['tweet'] = data['tweet'].apply(data_cleaning)
X = np.array(data["tweet"])
Y = np.array(data["labels"])

cv = CountVectorizer()  # convert text into number matrix to further systematically process it
X = cv.fit_transform(X)

# taking randomly 33 percent data from training set and giving it to testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# ML model - Decision Tree Classifier
dt = DecisionTreeClassifiersifier()
dt.fit(X_train, Y_train)

pred = dt.predict(X_test)
print("The accuracy of the model is: ",pred*100,"%")

cm = confusion_matrix(Y_test, pred)

# finding accuracy of the Ml model
accuracy_score(Y_test, pred)

# test a sample
example = "I will kill you if you hate someone"
example = data_cleaning(example)
matrix = cv.transform([example]).toarray()
final = dt.predict(matrix)
print("The given statement is: ",final)

