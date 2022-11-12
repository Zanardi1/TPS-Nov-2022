import os
from pathlib import Path

import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import log_loss
from tqdm import tqdm

path = Path('')

submission = pd.read_csv('sample_submission.csv', index_col='id')
labels = pd.read_csv('train_labels.csv', index_col='id')
sub_ids = submission.index
gt_ids = labels.index
subs = sorted(os.listdir('submission_files'))
scores_list = 0
processed_data = pd.DataFrame()

R = RidgeCV()

for i in tqdm(range(len(subs))):
    data = pd.read_csv(path / 'submission_files' / subs[i], index_col='id')
    score = log_loss(labels, data.loc[gt_ids])
    scores_list += data['pred']
    processed_data = pd.concat([processed_data, data['pred']], axis=1)

processed_data = processed_data.loc[gt_ids]
R.fit(processed_data, labels['label'])
print(R.score(processed_data, labels['label']))
blend = R.predict(processed_data)
print(log_loss(labels['label'], R.predict(processed_data)))
submission['pred'] = blend
submission.to_csv('blend.csv')
