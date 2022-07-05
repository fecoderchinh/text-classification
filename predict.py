from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
# from keras.layers import *

from pyvi import ViTokenizer, ViPosTagger
from tqdm import tqdm
import numpy as np
import gensim
import numpy as np

def preprocessing_doc(doc):
    lines = gensim.utils.simple_preprocess(doc)
    lines = ' '.join(lines)
    lines = ViTokenizer.tokenize(lines)

    return lines

import sys
import joblib
sys.modules['sklearn.externals.joblib'] = joblib

X_data = joblib.load(open('data/X_data.pkl', 'rb'))
y_data = joblib.load(open('data/y_data.pkl', 'rb'))

# X_test = pickle.load(open('data/X_test.pkl', 'rb'))
# y_test = pickle.load(open('data/y_test.pkl', 'rb'))

# word level - we choose max number of words equal to 30000 except all words (100k+ words)
tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
tfidf_vect.fit(X_data) # learn vocabulary and idf from training set
X_data_tfidf =  tfidf_vect.transform(X_data)

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=300, random_state=42)
svd.fit(X_data_tfidf)

X_data_tfidf_svd = svd.transform(X_data_tfidf)

encoder = preprocessing.LabelEncoder()
y_data_n = encoder.fit_transform(y_data)

from sklearn.model_selection import train_test_split
def train_model(classifier, X_data, y_data, X_test=None, y_test=None, is_neuralnet=False, n_epochs=3):
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42)

    if is_neuralnet:
        classifier.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs, batch_size=512)

        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(X_test)
        val_predictions = val_predictions.argmax(axis=-1)
    #         test_predictions = test_predictions.argmax(axis=-1)
    else:
        classifier.fit(X_train, y_train)

        train_predictions = classifier.predict(X_train)
        val_predictions = classifier.predict(X_val)
    #         test_predictions = classifier.predict(X_test)

    print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_val))
#     print("Test accuracy: ", metrics.accuracy_score(test_predictions, y_test))

model = naive_bayes.MultinomialNB()
train_model(model, X_data_tfidf, y_data, is_neuralnet=False)

import test as t
test_doc = preprocessing_doc(t.test_str())
# test_vec = get_word2vec_data([test_doc])

test_doc_tfidf = tfidf_vect.transform([test_doc])
# print(np.shape(test_doc_tfidf))
test_doc_svd = svd.transform(test_doc_tfidf)

print(model.predict(test_doc_tfidf))