import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from joblib import load, dump
from lime.lime_text import LimeTextExplainer

def prepare_data(train_dataset: pd.DataFrame):

    X_train, y_train= train_dataset["a"], train_dataset.iloc[:,-1]
    vectorizer = TfidfVectorizer(analyzer= 'word', stop_words='english')
    X_train_vector = vectorizer.fit_transform(X_train)
    dump(vectorizer, "./vectorizer")
        
    return X_train_vector, y_train

def prediction(new_text):
    train_dataset = pd.read_csv("./dataset/train_dataset.csv", encoding='latin-1')

    X_train_vector, y_train= prepare_data(train_dataset)
    vectorizer = load("./vectorizer")
    new_text_vector = vectorizer.transform([new_text])
    clf = RandomForestClassifier()
    clf.fit(X_train_vector, y_train)
    y_pred = clf.predict(new_text_vector)
    dump(clf, "./model.pkl")

    return y_pred



# dataset = pd.read_csv("./dataset/dataset.csv", encoding='latin-1')
# y = dataset.iloc[:,1]
# X = dataset['a']
# X_train, X_test, y_train, y_test = train_test_split(X , y ,test_size=0.2,shuffle=True)
# train_dataset = pd.DataFrame([X_train, y_train]).T
# test_dataset = pd.DataFrame([X_test, y_test]).T
# train_dataset.to_csv("./dataset/train_dataset.csv")
# test_dataset.to_csv("./dataset/test_dataset.csv")


# print(train_dataset)



# text = pd.read_csv("./dataset/test_dataset.csv", encoding='latin-1')
# text = text["a"][0]
# model = load('model.pkl')
# vectorizer = load("./vectorizer")
# classes = ["Colon_Cancer","Lung_Cancer","Thyroid_Cancer"]

# def predict_fn(text):
#     text_vector = vectorizer.transform(text)
#     return model.predict_proba(text_vector)

# explainer = LimeTextExplainer(class_names=classes)  
# explanation = explainer.explain_instance(text, predict_fn, num_features=5, top_labels=1)
# labels = explanation.available_labels()
# print(labels)

# for label in labels:
#     print(f"Explanation for class {label}:")
#     print('\n'.join(map(str, explanation.as_list(label=label))))
#     print()
# print(prediction(text))
