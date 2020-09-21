from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from datetime import  datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string

data = pd.read_csv('Add IMDB data location here')
game_data = pd.read_csv('Add chess games data location here')

model = SVC(C=100, gamma=0.001)


def custom_preprocessor(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)

    return text


def movie_data():
    data['review'] = data['review'].apply(custom_preprocessor)

    label_encoder = LabelEncoder()
    data['sentiment'] = label_encoder.fit_transform(data['sentiment'])

    data_X = data['review']
    data_y = data['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(data_X.fillna(0), data_y, test_size=0.2, random_state=0)

    cv = CountVectorizer()
    X_train_trans = cv.fit_transform(X_train)
    X_test_trans = cv.transform(X_test)

    start_time = timer(None)
    model.fit(X_train_trans, y_train)
    timer(start_time)

    prediction_y = model.predict(cv.transform(X_test))

    score_1 = accuracy_score(y_test, prediction_y)

    print(score_1)

    y_true, y_pred = y_test, model.predict(X_test_trans)
    print(classification_report(y_true, y_pred))

    fig, axes = plt.subplots(3, 2, figsize=(10, 15))

    title = r"Learning Curves (Multi-class Perceptron - Dataset 1)"
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = model
    plot_learning_curve(estimator, title, X_train_trans, y_train, axes=axes[:, 1], ylim=(0.7, 1.01),
                        cv=cv, n_jobs=4)

    plt.show()


def chess_game_data():
    data_X = game_data.drop(['winner', 'increment_code', 'white_id', 'black_id', 'created_at', 'last_move_at', 'id'],
                            axis=1)
    data_y = game_data[['winner']]

    X_train, X_test, y_train, y_test = train_test_split(data_X.fillna(0), data_y, test_size=0.2, random_state=0)

    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(X_train)

    X_train = encoder.transform(X_train)
    X_test = encoder.transform(X_test)

    start_time = timer(None)
    model.fit(X_train, y_train)
    timer(start_time)

    scores = cross_val_score(model, X_train, y_train, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    prediction = model.predict(X_test)

    print(accuracy_score(y_test, prediction))

    y_true, y_pred = y_test, model.predict(X_test)
    print(classification_report(y_true, y_pred))

    fig, axes = plt.subplots(3, 2, figsize=(10, 15))

    title = r"Learning Curves (Multi-class Perceptron - Dataset 2)"
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = model
    plot_learning_curve(estimator, title, X_train, y_train, axes=axes[:, 1], ylim=(0.7, 1.01),
                        cv=cv, n_jobs=4)

    plt.show()


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes, shuffle=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    return plt


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# movie_data()
chess_game_data()
