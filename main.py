import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.special
from colorama import Fore, Style
from sklearn.calibration import CalibrationDisplay
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline


def plot_oof_histogram(name, oof, title=None):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.hist(oof, range=(0, 1), bins=100, density=True, color='#ffd700')
    ax1.set_title(f'{name} prediction histogram')
    ax1.set_facecolor('#0057b8')
    CalibrationDisplay.from_predictions(y, oof, n_bins=20, strategy='quantile', ax=ax2, color='#ffd700')

    ax2.set_title('Probability calibration')
    ax2.set_facecolor('#0057b8')
    ax2.legend('', frameon=False)
    if title is not None:
        plt.suptitle(title, y=1, fontsize=20)
    plt.show()


def fun(x):
    return log_loss(y_tr, scipy.special.expit(scipy.special.logit(X_tr[:, [sub]]) - x))


X = np.zeros((40000, 5000))
names = sorted(glob.glob('submission_files\*.csv'))
for i, name in enumerate(names):
    sub = pd.read_csv(name)
    assert (sub.id == np.arange(40000)).all()
    if i == 0:
        print('Sample input file: ')
        print(sub)
    X[:, i] = sub.pred.values
    if i % 1000 == 0:
        print(i)

X_dev = X[:20000]
X_test = X[20000:]

y = pd.read_csv('train_labels.csv')
assert (y.id == np.arange(20000)).all()
print(y)
y = y.label.values

print(y.mean())

print('Minimum and maximum X values: ', X.min(), X.max())
print('Unique labels: ', np.unique(y))
print('Null values in X and y: ', np.isnan(X).sum(), np.isnan(y).sum())

print(f'Values below 0: {(X < 0).sum()}')
print(f'Values above 1: {(X > 1).sum()}')
print(f'Rows containing outliers: {((X < 0) | (X > 1)).any(axis=1).sum()} of {X.shape[0]}')
print(f'Columns containing outliers: {((X < 0) | (X > 1)).any(axis=0).sum()} of {X.shape[0]}')

print(np.unique(X, axis=0).shape)
print(np.unique(X, axis=1).shape)

print('Name logloss')
for sub in range(5):
    name_loss = float(names[sub][17:-4])
    print(names[sub][17:], name_loss, log_loss(y, X_dev[:, sub]))

log_loss_list = []
for sub in range(X.shape[1]):
    name_loss = float(names[sub][17:-4])
    log_loss_list.append(name_loss)
    assert abs(name_loss - log_loss(y, X_dev[:, sub])) < 1e-10

plt.hist(log_loss_list, bins=20, density=True, color='c')
plt.vlines([log_loss(y, np.full(len(y), 0.5))], 0, 30, color='r')
plt.text(0.693, 31, 'dummy', color='r', ha='center')
plt.title('Histogram of submission scores')
plt.xlabel('Log loss')
plt.ylabel('Density')
plt.ylim(0, 34)
plt.show()

for sub in range(X.shape[1]):
    if log_loss(y, X_dev[:, sub]) > log_loss(y, 1 - X_dev[:, sub]):
        print(
            f'Inverting improves column {sub} from {log_loss(y, X_dev[:, sub]):.4f} to {log_loss(y, 1 - X_dev[:, sub]):.4f}')

sample_subs = [0, 10, 20, 300, 393, 400, 500, 917, 922, 1451, 1461, 1818, 1834, 2009, 2600, 2811, 3023, 3560, 3851,
               4716, 4727, 4847, 4882, 4900]
_, axs = plt.subplots(6, 4, figsize=(16, 14))
axs = axs.ravel()
for i, sub in enumerate(sample_subs):
    axs[i].hist(X[:, sub], range=(0, 1), bins=100, density=True, color='#ffd700')
    axs[i].set_title(f'{sub}')
    axs[i].set_facecolor('#0057b8')
plt.tight_layout(h_pad=1)
plt.suptitle('Selected feature histograms', y=1.02, fontsize=20)
plt.show()

_, axs = plt.subplots(6, 4, figsize=(16, 14))
axs = axs.ravel()
for i, sub in enumerate(sample_subs):
    CalibrationDisplay.from_predictions(y, X_dev[:, sub].clip(0, 1), n_bins=20, strategy='quantile', ax=axs[i],
                                        color='#ffd700')
    axs[i].set_title(f'{sub}')
    axs[i].set_facecolor('#0057b8')
    axs[i].legend('', frameon=False)
plt.tight_layout(h_pad=1)
plt.suptitle('Selected probability calibrations', y=1.02, fontsize=20)
plt.show()

pca = PCA()
Xt = pca.fit_transform(X)
plt.plot(pca.explained_variance_ratio_.cumsum())
plt.title('Principal components analysis')
plt.xlabel('Component')
plt.ylabel('Cumulative explained variance ratio')
plt.show()

print('Cumulative explained variance ratio for the first 5 components: ',
      pca.explained_variance_ratio_.cumsum()[:5].round(2))

plt.figure(figsize=(8, 8))
plt.scatter(Xt[:20000, 0], Xt[:20000, 1], s=1, c=y)
plt.gca().set_aspect('equal')
plt.xlabel('First PCA component (90% of the total variance)')
plt.ylabel('Second PCA component (2% of the total variance)')
plt.title('PCA-projected labeled samples (train: yellow=1, blue=0)')
plt.show()

Xtt = pca.transform(X_test)
plt.figure(figsize=(8, 8))
plt.scatter(Xtt[:20000, 0], Xtt[:20000, 1], s=1, c='gray')
plt.gca().set_aspect('equal')
plt.xlabel('First PCA component (90% of the total variance)')
plt.ylabel('Second PCA component (2% of the total variance)')
plt.title('PCA-projected unlabeled samples (test)')
plt.show()

unique_count_list = []
for sub in range(X.shape[1]):
    unique_count = len(np.unique(X[:, sub]))
    unique_count_list.append(unique_count)
unique_counts = pd.Series(unique_count_list, name='unique')
unique_counts.sort_values(inplace=True)
print('Discrete submissions (few unique values):')
print(unique_counts.head(10))
print('Continuous submissions (many unique values)')
print(unique_counts.tail(10))

plt.hist(unique_counts, bins=np.linspace(0, 40000, 41))
plt.xlabel('Unique values')
plt.ylabel('Column count')
plt.title('Histogram of unique values per feature')
plt.show()

for sub in [590, 4727, 1853, 1814, 4855]:
    values = np.unique(X[:, sub])
    print(f'{Fore.GREEN}{Style.BRIGHT}Column {sub:4} unique values: {Style.RESET_ALL}{list(values)}')

overall_score_df = pd.DataFrame(columns=['logloss', 'oof'])

score_list = []
oof = np.zeros((len(X_dev),))
kf = StratifiedKFold(shuffle=True, random_state=1)
for fold, (idx_tr, idx_va) in enumerate(kf.split(X_dev, y)):
    X_tr = X_dev[idx_tr]
    X_va = X_dev[idx_va]
    y_tr = y[idx_tr]
    y_va = y[idx_va]

    model = make_pipeline(PCA(n_components=50), LogisticRegression(C=0.1, solver='saga'))
    model.fit(X_tr, y_tr)
    y_va_pred = model.predict_proba(X_va)[:, 1]
    logloss = log_loss(y_va, y_va_pred)
    oof[idx_va] = y_va_pred
    print(f'Fold {fold}: log_loss={logloss:.5f}')
    score_list.append(logloss)

print(f'{Fore.GREEN}{Style.BRIGHT} Average log loss: {sum(score_list) / len(score_list):.5f}{Style.RESET_ALL}')
overall_score_df.loc['LogisticRegression'] = (sum(score_list) / len(score_list), oof)
plot_oof_histogram('LogisticRegression', oof)

score_list = []
oof = np.zeros((len(X_dev),))
kf = StratifiedKFold(shuffle=True, random_state=1)
for fold, (idx_tr, idx_va) in enumerate(kf.split(X_dev, y)):
    X_tr = X_dev[idx_tr]
    X_va = X_dev[idx_va]
    y_tr = y[idx_tr]
    y_va = y[idx_va]

    model = make_pipeline(KNeighborsClassifier(n_neighbors=210))
    model.fit(X_tr, y_tr)
    y_va_pred = model.predict_proba(X_va)[:, 1]
    logloss = log_loss(y_va, y_va_pred)
    oof[idx_va] = y_va_pred
    print(f'Fold {fold}: log_loss={logloss:.5f}')
    score_list.append(logloss)

print(f'{Fore.GREEN}{Style.BRIGHT} Average log loss: {sum(score_list) / len(score_list):.5f}{Style.RESET_ALL}')
overall_score_df.loc['KNeighborsClassifier'] = (sum(score_list) / len(score_list), oof)
plot_oof_histogram('KNeighborsClassifier', oof)

for run in range(2):
    score_list = []
    oof = np.zeros((len(X_dev),))
    kf = StratifiedKFold(shuffle=True, random_state=1)
    for fold, (idx_tr, idx_va) in enumerate(kf.split(X_dev, y)):
        X_tr = X_dev[idx_tr]
        X_va = X_dev[idx_va]
        y_tr = y[idx_tr]
        y_va = y[idx_va]

    model = make_pipeline(ExtraTreesClassifier(n_estimators=100, min_samples_leaf=12, random_state=run))
    model.fit(X_tr, y_tr)
    y_va_pred = model.predict_proba(X_va)[:, 1]
    logloss = log_loss(y_va, y_va_pred)
    oof[idx_va] = y_va_pred
    print(f'Fold: {fold}, log_loss={logloss:.5f}')
    score_list.append(logloss)

print(f'{Fore.GREEN}{Style.BRIGHT} Average log loss: {sum(score_list) / len(score_list):.5f}{Style.RESET_ALL}')
overall_score_df.loc[f'ExtraTreesClassifier{run}'] = (sum(score_list) / len(score_list), oof)
plot_oof_histogram('ExtraTreesClassifier', oof)

score_list = []
oof = np.zeros((len(X_dev),))
kf = StratifiedKFold(shuffle=True, random_state=1)
for fold, (idx_tr, idx_va) in enumerate(kf.split(X_dev, y)):
    X_tr = X_dev[idx_tr]
    X_va = X_dev[idx_va]
    y_tr = y[idx_tr]
    y_va = y[idx_va]

    model = make_pipeline(PCA(80, random_state=1), MLPClassifier(learning_rate='adaptive', alpha=0.5, random_state=1))
    model.fit(X_tr, y_tr)
    y_va_pred = model.predict_proba(X_va)[:, 1]
    logloss = log_loss(y_va, y_va_pred)
    oof[idx_va] = y_va_pred
    print(f'Fold: {fold}, log_loss:{logloss:.5f}')
    score_list.append(logloss)

print(f'{Fore.GREEN}{Style.BRIGHT} Average log loss: {sum(score_list) / len(score_list):.5f}{Style.RESET_ALL}')
overall_score_df.loc[f'MLPClassifier'] = (sum(score_list) / len(score_list), oof)
plot_oof_histogram('MLPClassifier', oof)

plot_oof_histogram('Original', X_dev[:, 56], title='Before Calibration')
plot_oof_histogram('Calibrated',
                   IsotonicRegression(out_of_bounds='clip').fit_transform(X_dev[:, 56], y).clip(0.001, 0.999),
                   title='After calibration')

sub = 56
score_list = []
oof = np.zeros((len(X_dev),))
kf = StratifiedKFold(shuffle=True, random_state=1)
for fold, (idx_tr, idx_va) in enumerate(kf.split(X_dev, y)):
    X_tr = X_dev[idx_tr]
    X_va = X_dev[idx_va]
    y_tr = y[idx_tr]
    y_va = y[idx_va]

    model_isotonic = IsotonicRegression(out_of_bounds='clip')
    model_isotonic.fit(X_tr[:, sub], y_tr)
    y_va_pred = model_isotonic.predict(X_va[:, sub]).clip(0.001, 0.999)
    logloss = log_loss(y_va, y_va_pred)
    oof[idx_va] = y_va_pred
    print(f'Fold: {fold}, log_loss:{logloss:.5f}')
    score_list.append(logloss)

print(f'{Fore.GREEN}{Style.BRIGHT}Average log_loss {sub:4}: {sum(score_list) / len(score_list):.5f}{Style.RESET_ALL}')
overall_score_df.loc['IsotonicRegression'] = (sum(score_list) / len(score_list), oof)
plot_oof_histogram('IsotonicRegression', oof)

sub = 56
score_list = []
oof = np.zeros((len(X_dev),))
kf = StratifiedKFold(shuffle=True, random_state=1)
for fold, (idx_tr, idx_va) in enumerate(kf.split(X_dev, y)):
    X_tr = X_dev[idx_tr]
    X_va = X_dev[idx_va]
    y_tr = y[idx_tr]
    y_va = y[idx_va]

    delta = scipy.optimize.minimize(fun, x0=1, method='Nelder-Mead', options={'maxiter': 2000})
    delta = delta.x[0]

    y_va_pred = scipy.special.expit(scipy.special.logit(X_va[:, [sub]]) - delta)
    logloss = log_loss(y_va, y_va_pred)
    oof[idx_va] = y_va_pred.ravel()
    print(f'Fold: {sub}.{fold} - log_loss: {logloss:.5f}, delta: {delta:.3f}')
    score_list.append(logloss)

print(f'{Fore.GREEN}{Style.BRIGHT}Average log_loss {sub:4}:  {sum(score_list) / len(score_list):.5f}{Style.RESET_ALL}')
overall_score_df.loc['Logit-shift'] = (sum(score_list) / len(score_list), oof)
plot_oof_histogram('Logit-shift', oof)

x_plot = np.linspace(1e-6, 1 - 1e-6, 100)
plt.plot(x_plot, model_isotonic.predict(x_plot), label='Isotonic regression')
plt.plot(x_plot, scipy.special.expit(scipy.special.logit(x_plot) - delta), label='Logit shift')
plt.xlabel('Original probability')
plt.ylabel('Corrected probability')
plt.gca().set_aspect('equal')
plt.legend()
plt.title('Regression curves of isotonic regression and logit shift')
plt.show()

score_list = []
oof = np.zeros((len(X_dev),))
kf = StratifiedKFold(shuffle=True, random_state=1)
for fold, (idx_tr, idx_va) in enumerate(kf.split(X_dev, y)):
    X_tr = scipy.special.logit(X_dev[idx_tr].clip(1e-6, 1 - 1e-6))
    X_va = scipy.special.logit(X_dev[idx_va].clip(1e-6, 1 - 1e-6))
    y_tr = y[idx_tr]
    y_va = y[idx_va]

    model = make_pipeline(PCA(n_components=50, whiten=True, random_state=1),
                          LogisticRegression(C=0.1, solver='saga', random_state=1))
    model.fit(X_tr, y_tr)
    y_va_pred = model.predict_proba(X_va)[:, 1]
    logloss = log_loss(y_va, y_va_pred)
    oof[idx_va] = y_va_pred
    print(f"Fold {fold}: log_loss = {logloss:.5f}")
    score_list.append(logloss)

print(
    f'{Fore.GREEN}{Style.BRIGHT}Average log_loss:  {sum(score_list) / len(score_list):.5f}{Style.RESET_ALL}')  # 0.53180
overall_score_df.loc['LogisticRegression on logits'] = (sum(score_list) / len(score_list), oof)
plot_oof_histogram('LogisticRegression on logits', oof)

members = ['LogisticRegression on logits', 'ExtraTreesClassifier1']
members_oof = np.column_stack(overall_score_df.oof.loc[members])  # one column per ensemble member
score_list = []
oof = np.zeros((len(X_dev),))
kf = StratifiedKFold(shuffle=True, random_state=1)
for fold, (idx_tr, idx_va) in enumerate(kf.split(X_dev, y)):
    y_tr = y[idx_tr]
    y_va = y[idx_va]

    blend_model = Ridge(alpha=1e-1, fit_intercept=False)
    blend_model.fit(members_oof[idx_tr], y_tr)
    y_va_pred = blend_model.predict(members_oof[idx_va]).clip(0, 1)
    logloss = log_loss(y_va, y_va_pred)
    oof[idx_va] = y_va_pred
    print(f"Fold {fold}: log_loss = {logloss:.5f}   "
          f"weights = [{blend_model.coef_[0]:.2f} {blend_model.coef_[1]:.2f}]")
    score_list.append(logloss)

print(
    f'{Fore.GREEN}{Style.BRIGHT}Average log_loss:  {sum(score_list) / len(score_list):.5f}{Style.RESET_ALL}')  # 0.54517
overall_score_df.loc[f'Blend of {len(members)} models'] = (sum(score_list) / len(score_list), oof)
plot_oof_histogram('Blend', oof)

overall_score_df.sort_values('logloss', ascending=False, inplace=True)
plt.barh(np.arange(len(overall_score_df)), overall_score_df.logloss, color='chocolate')
plt.yticks(np.arange(len(overall_score_df)), overall_score_df.index)
plt.xlim(0.5, 0.56)
plt.title('Model comparison (shorter bar is better)')
plt.xlabel('log loss')
plt.show()
