## Cancer classification

Dataset for cancer classification was taken from kaggle [link](https://www.kaggle.com/datasets/falgunipatel19/biomedical-text-publication-classification) .

Dataset was divided into training and test datasets. Model was trained and deployed using Flask. On the website we can predict 3 types of cancer:
- colon cancer,
- lung cancer,
- thyroid cancer.

The homepage looks like this:

<p align="center">
  <img 
    width="800"
    height="500"
    src="https://user-images.githubusercontent.com/81253533/252488244-908ce3a7-51ac-4e6e-b896-65935aaee4b6.jpg"
  >
</p>

In order to predict cancer type, there is need to enter text in the form and submit. After submitting the following image will appear on the screen:

<p align="center">
  <img 
    width="800"
    height="500"
    src="https://user-images.githubusercontent.com/81253533/252488093-c097a164-d34a-4829-98b4-2b39f3af6bbe.jpg"
  >
</p>

In addition to the prediction itself, an explanation of the model's prediction was shown. This was done by [LIME](https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_text), a framework for explaining the predictions of machine learning models. It helps to understand and interpret the decisions made by the model on specific cases or examples.
