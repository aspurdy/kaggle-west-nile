from __future__ import print_function

from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

from src.preprocess2 import load_data


def grid(clf, tuned_parameters):
    (train, labels, test) = load_data()

    x_train, x_test, y_train, y_test = train_test_split(
        train, labels, test_size=0.5, random_state=0)

    kf = cross_validation.KFold(len(x_train), n_folds=4, shuffle=False, random_state=0)

    clf = GridSearchCV(clf, tuned_parameters, cv=kf, scoring='roc_auc', n_jobs=-1, verbose=1)
    clf.fit(x_train, y_train)
    print(clf.best_score_)
    print(clf.best_params_)
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))

    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true, y_pred))
    return clf
