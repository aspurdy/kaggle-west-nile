from sklearn import cross_validation

from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from xgboost import XGBClassifier
import pandas as pd
from preprocess2 import load_pkl

(train, labels, test) = load_pkl()
# train, labels = shuffle(train, labels, random_state=0)



valid_split = 3 * (len(train) / 4)

x_train = train[:valid_split, :]
x_valid = train[valid_split:, :]

y_train = labels[:valid_split]
y_valid = labels[valid_split:]

kf = KFold(len(train), n_folds=4, shuffle=False, random_state=0)
base = XGBClassifier()
clf = CalibratedClassifierCV(base, method='isotonic', cv=kf)

# clf.fit(x_train, y_train)
# print roc_auc_score(y_valid, clf.predict_proba(x_valid)[:, 1])
#
# scores = cross_validation.cross_val_score(base, train, labels, cv=kf, scoring='roc_auc')
# print("AUC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))


# calibrated xgb
clf.fit(train, labels)

df = pd.DataFrame()

df['WnvPresent'] = clf.predict_proba(test)[:, 1]
df.index += 1
df.to_csv('calibrated_xgb.csv', index=True, index_label='Id')

# uncalibrated xgb
base.fit(train, labels)
df = pd.DataFrame()

df['WnvPresent'] = base.predict_proba(test)[:, 1]
df.index += 1
df.to_csv('uncalibrated_xgb.csv', index=True, index_label='Id')
