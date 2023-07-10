from flask import Flask, render_template, request
from joblib import load
from lime.lime_text import LimeTextExplainer


app = Flask(__name__)
model = load('model.pkl')
classes = ["Colon_Cancer","Lung_Cancer","Thyroid_Cancer"]


@app.route('/', methods=["GET"])
def home():
    return render_template("index.html")


@app.route('/', methods=["POST"])
def predict():
    text = request.form['text']
    vectorizer = load("./vectorizer")
    text_vector = vectorizer.transform([text])
    pred = model.predict(text_vector)
    top_prediction = pred[0]

    def predict_fn(text):
        text_vector = vectorizer.transform(text)
        return model.predict_proba(text_vector)


    explainer = LimeTextExplainer(class_names=classes)
    explanation = explainer.explain_instance(text, predict_fn, num_features=5, top_labels=1)
    explanation_html = explanation.as_html()

    result = (top_prediction, explanation_html)
    return render_template("index.html", prediction=result)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
