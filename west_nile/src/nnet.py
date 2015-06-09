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
from sklearn import datasets
from sklearn.semi_supervised import LabelSpreading

from theano.tensor.nnet import sigmoid
from xgboost import XGBClassifier

from preprocess2 import load_pkl
from utils import AdjustVariable

(train, labels, test) = load_pkl()

input_size = len(train[0])

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

clf = NeuralNet(**nnet_params_1)
label_prop_model = LabelSpreading()

random_unlabeled_points = np.where(np.random.random_integers(0, 1, size=len(test)))
print random_unlabeled_points
# silver = np.copy(test)
# print silver.shape
# silver[random_unlabeled_points] = -1
# label_prop_model.fit(train, silver)
