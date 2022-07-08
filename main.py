import os

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from sklearn import naive_bayes, metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from pyvi import ViTokenizer
from sklearn.model_selection import train_test_split
from starlette.staticfiles import StaticFiles

from uvicorn import run

import sys
import joblib

sys.modules['sklearn.externals.joblib'] = joblib

app = FastAPI()

app.mount("/static", StaticFiles(directory="static", html=True), name="static")

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)

templates = Jinja2Templates(directory="templates")

# Uncomment these 2 lines if testing on locally
# X_data = joblib.load(open('data/X_data.pkl', 'rb'))
# y_data = joblib.load(open('data/y_data.pkl', 'rb'))
X_data = joblib.load(open('X_data_compressed.pkl', 'rb'))
y_data = joblib.load(open('y_data_compressed.pkl', 'rb'))

# word level - we choose max number of words equal to 30000 except all words (100k+ words)
tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
tfidf_vect.fit(X_data)  # learn vocabulary and idf from training set
X_data_tfidf = tfidf_vect.transform(X_data)
svd = TruncatedSVD(n_components=300, random_state=42)
svd.fit(X_data_tfidf)
model = naive_bayes.MultinomialNB()


def preprocessing_doc(doc):
    lines = gensim.utils.simple_preprocess(doc)
    lines = ' '.join(lines)
    lines = ViTokenizer.tokenize(lines)

    return lines


def train_model(classifier, X_data, y_data, X_test=None, y_test=None, is_neuralnet=False, n_epochs=3):
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42)

    if is_neuralnet:
        classifier.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs, batch_size=512)
        val_predictions = classifier.predict(X_val)
        val_predictions = val_predictions.argmax(axis=-1)
    else:
        classifier.fit(X_train, y_train)
        val_predictions = classifier.predict(X_val)

    return metrics.accuracy_score(val_predictions, y_val)


@app.exception_handler(404)
async def custom_404_handler(_, __):
    return RedirectResponse("/")


@app.exception_handler(405)
async def custom_405_handler(_, __):
    return RedirectResponse("/")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, 'message': 'Welcome to Text Classification API'})


@app.post("/predict", response_class=HTMLResponse)
async def get_text_prediction(request: Request, text: str = Form(...)):
    if text == '' or text is None:
        # return {"message": "No text provided"}
        return templates.TemplateResponse("predict.html", {"request": request, "message": "No text provided"})

    else:
        test_doc = preprocessing_doc(text)

        test_doc_tfidf = tfidf_vect.transform([test_doc])

        return templates.get_template('predict.html').render({
            "request": request,
            "text": text,
            "validation_accuracy": train_model(model, X_data_tfidf, y_data, is_neuralnet=False),
            "model_prediction": ', '.join(model.predict(test_doc_tfidf)),
        })


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 3000))
    run(app, host="0.0.0.0", port=port)
