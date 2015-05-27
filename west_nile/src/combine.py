__author__ = 'alex'

import pandas as pd
import numpy as np

sample = pd.read_csv('../input/sampleSubmission.csv')
gbc = pd.read_csv('combined/gbc.csv')
# lr = pd.read_csv('combined/lr.csv')
rfc = pd.read_csv('combined/rfc.csv')
svm = pd.read_csv('combined/svm.csv')

n = len(gbc)

df = gbc.merge(rfc, on='Id').merge(svm, on='Id')

df = (df.iloc[:, 1:].apply(np.log).sum(1) / 3).apply(np.exp)
sample['WnvPresent'] = df
sample.to_csv('combined/combined.csv', index=False)
gbc.update(sample)  # subset of cols = ['B']
