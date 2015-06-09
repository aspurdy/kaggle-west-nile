__author__ = 'alex'

import pandas as pd
import numpy as np
model1 = pd.read_csv('../predictions/1.csv')
model2 = pd.read_csv('../predictions/2.csv')
model3 = pd.read_csv('../predictions/3.csv')
sample = pd.read_csv('../input/sampleSubmission.csv')
# gbc = pd.read_csv('../src/gbc.csv')
# xgb = pd.read_csv('../src/xgb.csv')
# # lr = pd.read_csv('combined/lr.csv')
# # rfc = pd.read_csv('combined/rfc.csv')
# # svm = pd.read_csv('combined/svm.csv')
#
# n = len(gbc)
#
df = model1.merge(model2, on='Id').merge(model3, on='Id')
#
df = (df.iloc[:, 1:].apply(np.log).sum(1) / 3).apply(np.exp)
sample['WnvPresent'] = df
sample.to_csv('../src/combined.csv', index=False)
