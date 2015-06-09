import numpy as np
import pandas as pd
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer
from lasagne.updates import adagrad
from lasagne.objectives import binary_crossentropy
from lasagne.nonlinearities import rectify
from nolearn.lasagne import NeuralNet
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import shuffle
import theano
from theano import tensor as T

from theano.tensor.nnet import sigmoid
from xgboost import XGBClassifier

from preprocess2 import load_pkl
from utils import AdjustVariable

(train, labels, test) = load_pkl()



input_size = len(train[0])

xgb_params_1 = {'n_estimators': 100, 'subsample': 0.1, 'learning_rate': 0.01, 'max_depth': 3, 'gamma': 0}

gbt_params_1 = {
    'n_estimators': 100,
    'subsample': 1,
    'learning_rate': 0.1,
    'min_samples_split': 1,
    'max_depth': 1
}

labels = labels.reshape((-1, 1))
learning_rate = theano.shared(np.float32(0.1))

nnet_params_1 = {
    'layers': [
        ('input', InputLayer),
        ('hidden1', DenseLayer),
        ('dropout1', DropoutLayer),
        ('hidden2', DenseLayer),
        ('dropout2', DropoutLayer),
        ('output', DenseLayer),
    ],
    # layer parameters:
    'input_shape': (None, input_size),
    # 'hidden1_W': GlorotNormal(gain='relu'),
    'hidden1_num_units': 325,
    'hidden1_nonlinearity': rectify,
    'dropout1_p': 0.4,
    # 'hidden2_W': GlorotNormal(gain='relu'),
    'hidden2_nonlinearity': rectify,
    'hidden2_num_units': 325,
    'dropout2_p': 0.4,
    'output_nonlinearity': sigmoid,
    'output_num_units': 1,

    # optimization method:
    'update': adagrad,
    'update_learning_rate': learning_rate,
    # 'update_momentum': 0.9,

    # Decay the learning rate
    'on_epoch_finished': [
        AdjustVariable(learning_rate, target=0, half_life=4),
    ],

    # This is silly, but we don't want a stratified K-Fold here
    # To compensate we need to pass in the y_tensor_type and the loss.
    'regression': True,
    'y_tensor_type': T.lmatrix,
    'objective_loss_function': binary_crossentropy,

    'max_epochs': 75,
    'eval_size': 0.1,
    'verbose': 0,
}

clf1 = XGBClassifier(**xgb_params_1)
clf2 = GradientBoostingClassifier(**gbt_params_1)
clf3 = NeuralNet(**nnet_params_1)
clfs=[clf1, clf2, clf3]
df = pd.DataFrame()
for (i, clf) in enumerate(clfs):
    print i
    if i == 2:
        train, labels = shuffle(train, labels, random_state=0)
    clf.fit(train, labels)
    pred = clf.predict_proba(test)
    if pred.shape[1] > 1:
        pred = pred[:, 1]
    df['%s' % clf] = pred

out = pd.DataFrame()
n_classifiers = df.shape[1]

# geometric average
# out['WnvPresent'] = (df.iloc[:, 1:].apply(np.log).sum(1) / n_classifiers).apply(np.exp)
out['WnvPresent'] = (df.iloc[:, 1:].apply(np.log).sum(1) / n_classifiers).apply(np.exp)
out.index += 1
out.to_csv('combined.csv', index=True, index_label='Id')

# train, labels = shuffle(train, labels, random_state=0)

#
# weights = []
#
# kf = KFold(len(train), n_folds=4, shuffle=False, random_state=0)
# for (train_idx, test_idx) in kf:
#     df = pd.DataFrame()
#     for clf in clfs:
#         clf.fit(train[train_idx], labels[train_idx])
#         pred = clf.predict_proba(train[test_idx])
#         if pred.shape[1] == 1:
#             df['%s' % clf] = clf.predict_proba(train[test_idx])
#         else:
#             df['%s' % clf] = clf.predict_proba(train[test_idx])[:, 1]
#     clf = GradientBoostingClassifier(random_state=0).fit(df, labels[test_idx])
#     weights.append(clf.feature_importances_)
#
# weights = np.array(weights)
# print("Mean weights: %0.2f (+/- %0.2f)" % (weights.mean(), weights.std() / 2))



# for clf in clfs:
#     clf.fit(train, labels)
#
# preds = [clf.predict_proba(test) for clf in clfs]
# for pred in preds:
#     print pred.shape
# print preds
# df = pd.DataFrame()
# df['WnvPresent'] =
# df.index += 1
# df.to_csv('ensemble.csv', index=True, index_label='Id')
