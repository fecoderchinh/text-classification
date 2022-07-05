# Các gói thư viện phục vụ tiền xử lý dữ liệu
from keras.layers import Bidirectional
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

# Thư viện phục vụ chuẩn bị dữ liệu

from pyvi import ViTokenizer, ViPosTagger
from tqdm import tqdm
import numpy as np
import gensim
import numpy as np

# Hàm get_data()
import os
import sys
import joblib

sys.modules['sklearn.externals.joblib'] = joblib
dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
dir_path = os.path.join(dir_path, 'text-classification/data')


def get_data(folder_path):
    X = []
    y = []
    dirs = os.listdir(folder_path)
    for path in dirs:
        file_paths = os.listdir(os.path.join(folder_path, path))
        for file_path in tqdm(file_paths):
            with open(os.path.join(folder_path, path, file_path), 'r', encoding="utf-16") as f:
                lines = f.readlines()
                lines = ' '.join(lines)
                lines = gensim.utils.simple_preprocess(lines)
                lines = ' '.join(lines)
                lines = ViTokenizer.tokenize(lines)
                #                 sentence = ' '.join(words)
                #                 print(lines)
                X.append(lines)
                y.append(path)
    #             break
    #         break
    return X, y


if not os.path.exists(os.path.join(dir_path, 'X_data.pkl')) and not os.path.exists(
        os.path.join(dir_path, 'y_data.pkl')):
    print("Data files not found, build it!")
    train_path = os.path.join(dir_path, 'Train_Full')
    X_data, y_data = get_data(train_path)

    joblib.dump(X_data, open('data/X_data.pkl', 'wb'))
    joblib.dump(y_data, open('data/y_data.pkl', 'wb'))

else:
    X_data = joblib.load(open('data/X_data.pkl', 'rb'))
    y_data = joblib.load(open('data/y_data.pkl', 'rb'))
    print("Data files have been found and loaded.")

if not os.path.exists(os.path.join(dir_path, 'X_test.pkl')) and not os.path.exists(
        os.path.join(dir_path, 'y_test.pkl')):
    print("Test files not found, build it!")
    test_path = os.path.join(dir_path, 'Test_Full')
    X_test, y_test = get_data(test_path)

    joblib.dump(X_test, open('data/X_test.pkl', 'wb'))
    joblib.dump(y_test, open('data/y_test.pkl', 'wb'))

else:
    X_test = joblib.load(open('data/X_test.pkl', 'rb'))
    y_test = joblib.load(open('data/y_test.pkl', 'rb'))
    print("Test files have been found and loaded.")

# Count Vectors as features

# create a count vectorizer object
print('start creating a count vectorizer object')
count_vector = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vector.fit(X_data)

# transform the training and validation data using count vectorizer object
X_data_count = count_vector.transform(X_data)
X_test_count = count_vector.transform(X_test)

# TF-IDF Vectors

print('start TfidfVectorizer')
# word level - we choose max number of words equal to 30000 except all words (100k+ words)
tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
tfidf_vect.fit(X_data)  # learn vocabulary and idf from training set
X_data_tfidf = tfidf_vect.transform(X_data)
# assume that we don't have test set before
X_test_tfidf = tfidf_vect.transform(X_test)

tfidf_vect.get_feature_names_out()

print('start TfidfVectorizer ngram level')
# ngram level - we choose max number of words equal to 30000 except all words (100k+ words)
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', max_features=30000, ngram_range=(2, 3))
tfidf_vect_ngram.fit(X_data)
X_data_tfidf_ngram = tfidf_vect_ngram.transform(X_data)
# assume that we don't have test set before
X_test_tfidf_ngram = tfidf_vect_ngram.transform(X_test)

tfidf_vect_ngram.get_feature_names_out()

print('start TfidfVectorizer ngram-char level')
# ngram-char level - we choose max number of words equal to 30000 except all words (100k+ words)
tfidf_vect_ngram_char = TfidfVectorizer(analyzer='char', max_features=30000, ngram_range=(2, 3))
tfidf_vect_ngram_char.fit(X_data)
X_data_tfidf_ngram_char = tfidf_vect_ngram_char.transform(X_data)
# assume that we don't have test set before
X_test_tfidf_ngram_char = tfidf_vect_ngram_char.transform(X_test)

# Transform by SVD to decrease number of dimensions
# Word Level
from sklearn.decomposition import TruncatedSVD

print('start TruncatedSVD')
svd = TruncatedSVD(n_components=300, random_state=42)
svd.fit(X_data_tfidf)
X_data_tfidf_svd = svd.transform(X_data_tfidf)
X_test_tfidf_svd = svd.transform(X_test_tfidf)

print('start TruncatedSVD ngram Level')
# ngram Level
svd_ngram = TruncatedSVD(n_components=300, random_state=42)
svd_ngram.fit(X_data_tfidf_ngram)

X_data_tfidf_ngram_svd = svd_ngram.transform(X_data_tfidf_ngram)
X_test_tfidf_ngram_svd = svd_ngram.transform(X_test_tfidf_ngram)

# ngram Char Level
print('start TruncatedSVD ngram Char Level')
svd_ngram_char = TruncatedSVD(n_components=300, random_state=42)
svd_ngram_char.fit(X_data_tfidf_ngram_char)
X_data_tfidf_ngram_char_svd = svd_ngram_char.transform(X_data_tfidf_ngram_char)
X_test_tfidf_ngram_char_svd = svd_ngram_char.transform(X_test_tfidf_ngram_char)

# Word Embeddings
from gensim.models import KeyedVectors

print('start KeyedVectors')
dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
word2vec_model_path = os.path.join(dir_path, "text-classification/data/vi/vi.vec")

w2v = KeyedVectors.load_word2vec_format(word2vec_model_path)
vocab = w2v.index_to_key
wv = w2v


def get_word2vec_data(X):
    word2vec_data = []
    for x in X:
        sentence = []
        for word in x.split(" "):
            if word in vocab:
                #                 print(word)
                sentence.append(wv[word])

        word2vec_data.append(sentence)
    #         break
    return word2vec_data


X_data_w2v = get_word2vec_data(X_data)
X_test_w2v = get_word2vec_data(X_test)

print('start LabelEncoder')
encoder = preprocessing.LabelEncoder()
y_data_n = encoder.fit_transform(y_data)
y_test_n = encoder.fit_transform(y_test)

encoder.classes_  # kết quả: array(['Chinh tri Xa hoi', 'Doi song', 'Khoa hoc', 'Kinh doanh',
#                 'Phap luat', 'Suc khoe', 'The gioi', 'The thao', 'Van hoa',
#                 'Vi tinh'], dtype='<U16')


from sklearn.model_selection import train_test_split


def train_model(classifier, X_data, y_data, X_test, y_test, is_neuralnet=False, n_epochs=3):
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42)

    if is_neuralnet:
        classifier.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs, batch_size=512)

        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(X_test)
        val_predictions = val_predictions.argmax(axis=-1)
        test_predictions = test_predictions.argmax(axis=-1)
    else:
        classifier.fit(X_train, y_train)

        train_predictions = classifier.predict(X_train)
        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(X_test)

    print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_val))
    print("Test accuracy: ", metrics.accuracy_score(test_predictions, y_test))


print('start train_model MultinomialNB')
train_model(naive_bayes.MultinomialNB(), X_data_tfidf, y_data, X_test_tfidf, y_test, is_neuralnet=False)

# train_model(naive_bayes.MultinomialNB(), X_data_tfidf_ngram_svd, y_data, X_test_tfidf_ngram_svd, y_test, is_neuralnet=False)

# train_model(naive_bayes.MultinomialNB(), X_data_tfidf_ngram_char_svd, y_data, X_test_tfidf_ngram_char_svd, y_test, is_neuralnet=False)

print('start train_model BernoulliNB')
train_model(naive_bayes.BernoulliNB(), X_data_tfidf, y_data, X_test_tfidf, y_test, is_neuralnet=False)

print('start train_model BernoulliNB svd')
train_model(naive_bayes.BernoulliNB(), X_data_tfidf_svd, y_data, X_test_tfidf_svd, y_test, is_neuralnet=False)

# Linear Classifier
print('start train_model LogisticRegression')
train_model(linear_model.LogisticRegression(), X_data_tfidf, y_data, X_test_tfidf, y_test, is_neuralnet=False)
print('start train_model LogisticRegression svd')
train_model(linear_model.LogisticRegression(), X_data_tfidf_svd, y_data, X_test_tfidf_svd, y_test, is_neuralnet=False)

# SVM Model
print('start train_model SVM')
train_model(svm.SVC(), X_data_tfidf_svd, y_data, X_test_tfidf_svd, y_test, is_neuralnet=False)

# Bagging Model
print('start train_model RandomForestClassifier')
train_model(ensemble.RandomForestClassifier(), X_data_tfidf_svd, y_data, X_test_tfidf_svd, y_test, is_neuralnet=False)

# Boosting Model
print('start train_model XGBClassifier')
train_model(xgboost.XGBClassifier(), X_data_tfidf_svd, y_data, X_test_tfidf_svd, y_test, is_neuralnet=False)

# Deep Neural Network
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import models


def create_dnn_model():
    input_layer = Input(shape=(300,))
    layer = Dense(1024, activation='relu')(input_layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dense(512, activation='relu')(layer)
    output_layer = Dense(10, activation='softmax')(layer)

    classifier = models.Model(input_layer, output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return classifier


print('start classifing dnn')
classifier = create_dnn_model()
train_model(classifier=classifier, X_data=X_data_tfidf_svd, y_data=y_data_n, X_test=X_test_tfidf_svd, y_test=y_test_n,
            is_neuralnet=True)


# Recurrent Neural Network

def create_lstm_model():
    input_layer = Input(shape=(300,))

    layer = Reshape((10, 30))(input_layer)
    layer = LSTM(128, activation='relu')(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dense(128, activation='relu')(layer)

    output_layer = Dense(10, activation='softmax')(layer)

    classifier = models.Model(input_layer, output_layer)

    classifier.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return classifier


print('start classifing lstm')
classifier = create_lstm_model()
train_model(classifier=classifier, X_data=X_data_tfidf_svd, y_data=y_data_n, X_test=X_test_tfidf_svd, y_test=y_test_n,
            is_neuralnet=True)


def create_gru_model():
    input_layer = Input(shape=(300,))

    layer = Reshape((10, 30))(input_layer)
    layer = GRU(128, activation='relu')(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dense(128, activation='relu')(layer)

    output_layer = Dense(10, activation='softmax')(layer)

    classifier = models.Model(input_layer, output_layer)

    classifier.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return classifier


print('start classifing gru')
classifier = create_gru_model()
train_model(classifier=classifier, X_data=X_data_tfidf_svd, y_data=y_data_n, X_test=X_test_tfidf_svd, y_test=y_test_n,
            is_neuralnet=True, n_epochs=10)


def create_brnn_model():
    input_layer = Input(shape=(300,))

    layer = Reshape((10, 30))(input_layer)
    layer = Bidirectional(GRU(128, activation='relu'))(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dense(128, activation='relu')(layer)

    output_layer = Dense(10, activation='softmax')(layer)

    classifier = models.Model(input_layer, output_layer)

    classifier.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return classifier


print('start classifing brnn')
classifier = create_brnn_model()
train_model(classifier=classifier, X_data=X_data_tfidf_svd, y_data=y_data_n, X_test=X_test_tfidf_svd, y_test=y_test_n,
            is_neuralnet=True, n_epochs=20)


def create_rcnn_model():
    input_layer = Input(shape=(300,))

    layer = Reshape((10, 30))(input_layer)
    layer = Bidirectional(GRU(128, activation='relu', return_sequences=True))(layer)
    layer = Convolution1D(100, 3, activation="relu")(layer)
    layer = Flatten()(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dense(128, activation='relu')(layer)

    output_layer = Dense(10, activation='softmax')(layer)

    classifier = models.Model(input_layer, output_layer)
    classifier.summary()
    classifier.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return classifier


print('start classifing rcnn')
classifier = create_rcnn_model()
train_model(classifier=classifier, X_data=X_data_tfidf_svd, y_data=y_data_n, X_test=X_test_tfidf_svd, y_test=y_test_n,
            is_neuralnet=True, n_epochs=20)


def get_corpus(documents):
    corpus = []

    for i in tqdm(range(len(documents))):
        doc = documents[i]

        words = doc.split(' ')
        tagged_document = gensim.models.doc2vec.TaggedDocument(words, [i])

        corpus.append(tagged_document)

    return corpus

print('start trainning corpus')
train_corpus = get_corpus(X_data)

test_corpus = get_corpus(X_test)

model = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=2, epochs=40)
model.build_vocab(train_corpus)

import time

start = time.time()
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
end = time.time()
print(end - start)

X_data_vectors = []
for x in train_corpus:
    vector = model.infer_vector(x.words)
    X_data_vectors.append(vector)

X_test_vectors = []
for x in test_corpus:
    vector = model.infer_vector(x.words)
    X_test_vectors.append(vector)

# classifier = create_dnn_model()
# train_model(classifier=classifier, X_data=np.array(X_data_vectors), y_data=y_data_n, X_test=(X_test_vectors), y_test=y_test_n, is_neuralnet=True, n_epochs=5)
