import bz2

from pyvi import ViTokenizer, ViPosTagger  # thư viện NLP tiếng Việt
from tqdm import tqdm
import gensim  # thư viện NLP
import os
import sys
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

sys.modules['sklearn.externals.joblib'] = joblib
dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
dir_path = os.path.join(dir_path, 'text-classification/data')


def get_data(folder_path):
    X = []
    y = []
    dirs = os.listdir(folder_path)
    for path in tqdm(dirs):
        file_paths = os.listdir(os.path.join(folder_path, path))
        for file_path in tqdm(file_paths):
            with open(os.path.join(folder_path, path, file_path), 'r', encoding="utf-16") as f:
                lines = f.readlines()
                lines = ' '.join(lines)
                lines = gensim.utils.simple_preprocess(lines)
                lines = ' '.join(lines)
                lines = ViTokenizer.tokenize(lines)

                X.append(lines)
                y.append(path)

    return X, y


if not os.path.exists(os.path.join(dir_path, 'X_data_compressed.pkl')) and not os.path.exists(
        os.path.join(dir_path, 'y_data_compressed.pkl')):
    print("Data files not found, build it!")
    train_path = os.path.join(dir_path, 'Train_Full')
    X_data, y_data = get_data(train_path)

    # joblib.dump(X_data, open('data/X_data.pkl', 'wb'))
    # joblib.dump(y_data, open('data/y_data.pkl', 'wb'))
    joblib.dump(X_data, bz2.BZ2File('data/X_data_compressed.pkl', 'wb'))
    joblib.dump(y_data, bz2.BZ2File('data/y_data_compressed.pkl', 'wb'))

else:
    # X_data = joblib.load(open('data/X_data.pkl', 'rb'))
    # y_data = joblib.load(open('data/y_data.pkl', 'rb'))
    X_data = joblib.load(bz2.BZ2File('data/X_data_compressed.pkl', 'rb'))
    y_data = joblib.load(bz2.BZ2File('data/y_data_compressed.pkl', 'rb'))
    print("Data files have been found and loaded.")

if not os.path.exists(os.path.join(dir_path, 'X_test_compressed.pkl')) and not os.path.exists(
        os.path.join(dir_path, 'y_test_compressed.pkl')):
    print("Test files not found, build it!")
    test_path = os.path.join(dir_path, 'Test_Full')
    X_test, y_test = get_data(test_path)

    # joblib.dump(X_test, open('data/X_test.pkl', 'wb'))
    # joblib.dump(y_test, open('data/y_test.pkl', 'wb'))
    joblib.dump(X_test, bz2.BZ2File('data/X_test_compressed.pkl', 'wb'))
    joblib.dump(y_test, bz2.BZ2File('data/y_test_compressed.pkl', 'wb'))

else:
    # X_test = joblib.load(open('data/X_test.pkl', 'rb'))
    # y_test = joblib.load(open('data/y_test.pkl', 'rb'))
    X_test = joblib.load(bz2.BZ2File('data/X_test_compressed.pkl', 'rb'))
    y_test = joblib.load(bz2.BZ2File('data/y_test_compressed.pkl', 'rb'))
    print("Test files have been found and loaded.")
