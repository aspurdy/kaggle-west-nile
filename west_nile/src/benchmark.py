from __future__ import print_function

import pandas as pd
import xgboost as xgb
from sklearn import cross_validation

from src.preprocess2 import load_data

(train, labels, test) = load_data()

# drop columns with -1s
# train = train.ix[:, (train != -1).any(axis=0)]
# test = test.ix[:, (test != -1).any(axis=0)]

kf = cross_validation.KFold(len(train), n_folds=4, shuffle=False, random_state=0)
# params = {
#     'n_estimators': 1000,
#     'min_samples_split': 1
# }
#
#
# clf = ensemble.RandomForestClassifier(n_jobs=-1, **params)
# scores = cross_validation.cross_val_score(clf, train, labels, cv=kf, scoring='roc_auc')
# print("AUC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))
#
# # for split in [0.3, 0.5, 0.7]:
# #     x_train, x_test, y_train, y_test = train_test_split(train, labels, test_size=split)
# #     clf.fit(x_train, y_train)
# #     print("HO for split %0.2f AUC: %0.2f" % (split, roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])))
#
#
#
# # clf.fit(train, labels)
# # rfc_predictions = clf.predict_proba(test)[:, 1]
#
#
# tuned_params = {
#     'n_estimators': 100,
#     'subsample': 1,
#     'learning_rate': 0.1,
#     'min_samples_split': 1,
#     'max_depth': 1
# }
#
# clf = ensemble.GradientBoostingClassifier(**tuned_params)
# scores = cross_validation.cross_val_score(clf, train, labels, cv=kf, scoring='roc_auc')
# print("AUC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))
#
# df = pd.DataFrame()
# clf.fit(train, labels)
#
# df['WnvPresent'] = clf.predict_proba(test)[:, 1]
# df.index += 1
# df.to_csv('gbc.csv', index=True, index_label='Id')

#
# #
# # for split in [0.3, 0.5, 0.7]:
# #     x_train, x_test, y_train, y_test = train_test_split(train, labels, test_size=split)
# #     clf.fit(x_train, y_train)
# #     print("HO for split %0.2f AUC: %0.2f" % (split, roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])))
# #
#
#
# # clf.fit(train, labels)
# # create predictions and submission file
# # gbc_predictions = clf.predict_proba(test)[:, 1]
#
# params = {
#     'penalty': 'l2',
#     'C': 1e5,
# }
#
# clf = linear_model.LogisticRegression(**params)
# scores = cross_validation.cross_val_score(clf, train, labels, cv=kf, scoring='roc_auc')
# print("AUC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))
#
# # for split in [0.3, 0.5, 0.7]:
# #     x_train, x_test, y_train, y_test = train_test_split(train, labels, test_size=split)
# #     clf.fit(x_train, y_train)
# #     print("HO for split %0.2f AUC: %0.2f" % (split, roc_auc_score(y_test, clf.predict_proba(x_test.astype(float))[:, 1])))
# # lr_predictions = clf.predict_proba(test.astype(float))[:, 1]
#
# # fixed_params = {
# #     # 'kernel': 'rbf',
# #     # 'gamma': 0.0,
# #     'probability': True,
# #     'class_weight': 'auto',
# #     # 'verbose': 1
# # }
#
# # tuned_parameters = [
# #     {
# #         'kernel': ['rbf'],
# #         'gamma': [1e-3, 1e-4, 1e-5],
# #         'C': [1, 10, 100, 1000],
# #         'class_weight': [None, 'auto'],
# #     },
# # ]
# #
# #
# # for split in [0.3]:
# #     x_train, x_test, y_train, y_test = train_test_split(train, labels, test_size=split)
# #     clf = GridSearchCV(SVC(**fixed_params), tuned_parameters, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
# #     clf.fit(x_train, y_train)
# #     print("Best parameters set found on development set:\n")
# #     print(clf.best_params_)
# #
# #     for params, mean_score, scores in clf.grid_scores_:
# #         print("%0.3f (+/-%0.03f) for %r"
# #               % (mean_score, scores.std() * 2, params))
# #     print("Detailed classification report:\n")
# #     print("The model is trained on the full development set.")
# #     print("The scores are computed on the full evaluation set.\n")
# #     print("HO for split %0.2f AUC: %0.2f" % (split, roc_auc_score(y_test, clf.predict_proba(x_test.astype(float))[:, 1])))
#
# params = {
#     'kernel': 'rbf',
#     'C': 1000,
#     'gamma': 1e-05,
#     'probability': True,
#     'class_weight': 'auto',
#     # 'verbose': 1
# }
#
# clf = SVC(**params)
# scores = cross_validation.cross_val_score(clf, train, labels, cv=kf, scoring='roc_auc')
# print("AUC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))

tuned_params = {
    'n_estimators': [100, 1000],
    'subsample': [0.1, 0.5, 1],
    'learning_rate': [1, 0.1, 0.01],
    'max_depth': [1, 2, 3],
    'gamma': [0, 1, 10],
}
#
# fixed_params = {
#     'missing': np.NaN
# }
# best_params = {'n_estimators': 1000, 'subsample': 0.5, 'learning_rate': 0.01, 'max_depth': 3, 'gamma': 1}
best_params = {'n_estimators': 1000, 'subsample': 0.1, 'learning_rate': 0.01, 'max_depth': 3, 'gamma': 0}

clf = xgb.XGBClassifier(**best_params)
# clf = grid(xgb.XGBClassifier(), tuned_params)

scores = cross_validation.cross_val_score(clf, train, labels, cv=kf, scoring='roc_auc')
print("AUC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))

clf.fit(train, labels)
df = pd.DataFrame()

df['WnvPresent'] = clf.predict_proba(test)[:, 1]
df.index += 1
df.to_csv('xgb.csv', index=True, index_label='Id')

# clf.fit(train, labels)
# svm_predictions = clf.predict_proba(test.astype(float))[:, 1]

# pred_dict = {'rfc': rfc_predictions, 'gbc': gbc_predictions, 'lr': lr_predictions, 'svm': svm_predictions}
# # pred_dict = {'svm': svm_predictions}
# for key in pred_dict:
#     sample['WnvPresent'] = pred_dict[key]
#     sample.to_csv('combined/%s.csv' % key, index=False)
