from ucimlrepo import fetch_ucirepo 
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from imblearn.utils._validation import _count_class_sample
import random

dataset_name_id = dict(wine=186, musk2=75, vehicle=149, wave=107, isolet=54, abalone=1, yeast=110, bank=222)

dataset_positive_label=dict(wine=7, musk2=1, letter='U', satellite=5, vehicle='van', wave=1, isolet=1.0)
# dataset_positive_label.update(abalone=11, yeast="ME2", bank='yes', income='>50K', spam=1, dota2=-1)
# dataset_positive_label.update(student='Dropout')

synth_data = {
              'synth_1':dict(n_samples=10000, n_features=10, n_classes=2, weights=[0.9,0.1], n_informative=10, n_redundant=0, random_state=42),
              'synth_2':dict(n_samples=10000, n_features=20, n_classes=2, weights=[0.9,0.1], n_informative=20, n_redundant=0, random_state=42),
              'synth_3':dict(n_samples=10000, n_features=30, n_classes=2, weights=[0.9,0.1], n_informative=30, n_redundant=0, random_state=42),
              'synth_4':dict(n_samples=10000, n_features=40, n_classes=2, weights=[0.9,0.1], n_informative=40, n_redundant=0, random_state=42),
              'synth_5':dict(n_samples=10000, n_features=50, n_classes=2, weights=[0.9,0.1], n_informative=50, n_redundant=0, random_state=42),
              'synth_6':dict(n_samples=10000, n_features=60, n_classes=2, weights=[0.9,0.1], n_informative=60, n_redundant=0, random_state=42),
              'synth_7':dict(n_samples=10000, n_features=70, n_classes=2, weights=[0.9,0.1], n_informative=70, n_redundant=0, random_state=42),
              'synth_8':dict(n_samples=10000, n_features=80, n_classes=2, weights=[0.9,0.1], n_informative=80, n_redundant=0, random_state=42),
              'synth_9':dict(n_samples=10000, n_features=90, n_classes=2, weights=[0.9,0.1], n_informative=90, n_redundant=0, random_state=42),
              'synth_10':dict(n_samples=10000, n_features=100, n_classes=2, weights=[0.9,0.1], n_informative=100, n_redundant=0, random_state=42),
            }

synth_data_1 = {
              'synth_11':dict(n_samples=1000, n_features=50, n_classes=2, weights=[0.9,0.1], n_informative=50, n_redundant=0, random_state=42),
              'synth_12':dict(n_samples=2000, n_features=50, n_classes=2, weights=[0.9,0.1], n_informative=50, n_redundant=0, random_state=42),
              'synth_13':dict(n_samples=4000, n_features=50, n_classes=2, weights=[0.9,0.1], n_informative=50, n_redundant=0, random_state=42),
              'synth_14':dict(n_samples=8000, n_features=50, n_classes=2, weights=[0.9,0.1], n_informative=50, n_redundant=0, random_state=42),
              'synth_15':dict(n_samples=16000, n_features=50, n_classes=2, weights=[0.9,0.1], n_informative=50, n_redundant=0, random_state=42),
            }

synth_data_2 = {
              'synth_21':dict(n_samples=8000, n_features=20, n_classes=2, weights=[0.5,0.5], n_informative=20, n_redundant=0, random_state=42),
              'synth_22':dict(n_samples=8000, n_features=20, n_classes=2, weights=[0.7,0.3], n_informative=20, n_redundant=0, random_state=42),
              'synth_23':dict(n_samples=8000, n_features=20, n_classes=2, weights=[0.8,0.2], n_informative=20, n_redundant=0, random_state=42),
              'synth_24':dict(n_samples=8000, n_features=20, n_classes=2, weights=[0.9,0.1], n_informative=20, n_redundant=0, random_state=42),
              'synth_25':dict(n_samples=8000, n_features=20, n_classes=2, weights=[0.95,0.05], n_informative=20, n_redundant=0, random_state=42),
              'synth_26':dict(n_samples=8000, n_features=20, n_classes=2, weights=[0.98,0.02], n_informative=20, n_redundant=0, random_state=42),
              'synth_27':dict(n_samples=8000, n_features=20, n_classes=2, weights=[0.99,0.01], n_informative=20, n_redundant=0, random_state=42),
            }

dataset_dir = './dataset'

def load_UCI_data(data, dataset_name=None):
    if not os.path.exists(os.path.join(dataset_dir, f'{dataset_name}.pkl')):
        dataset = fetch_ucirepo(id=data[dataset_name]) # fetch dataset 
        X = dataset.data.features
        target = dataset.data.targets
        if 'income' in target.columns:  # income dataset clean target
            target['income'] = target['income'].apply(lambda x: x.replace('.','') if type(x)==str else x)

        with open(os.path.join(dataset_dir, f'{dataset_name}.pkl'), 'wb') as f:
            pickle.dump([X, target], f)
    else:
        with open(os.path.join(dataset_dir, f'{dataset_name}.pkl'), 'rb') as f:
            X, target = pickle.load(f)

    target_count = target.value_counts()
    # print(target)
    print(target_count)
    
    positive_target = dataset_positive_label[dataset_name]
    print(positive_target)

    X = X.dropna(axis=0, how='any')
    X = pd.get_dummies(X, dtype=float)  # one-hot for catagorial feature
    X = (X - X.min())/(X.max() - X.min())  # scale to 0~1, not need for tree model
    
    y = np.where(target == positive_target, 1, 0)
    y = y.reshape((y.shape[0],))
    
    ratio = sum(y==0) / sum(y==1) 
    print(f'this dataset imbalance ratio is:{ratio}') 

    # print(X.describe())
    X = np.array(X)
    print(X.shape)
    
    return X, y


def synthetic_data(data, dataset_name):
    dataset_config = data.get(dataset_name)
    if not os.path.exists(os.path.join(dataset_dir, f'{dataset_name}.pkl')):
        X, y = make_classification(**dataset_config)
        X = (X - X.min())/(X.max() - X.min())  # scale to 0~1, not need for tree model
        with open(os.path.join(dataset_dir, f'{dataset_name}.pkl'), 'wb') as f:
                pickle.dump([X, y], f)
    else:
        with open(os.path.join(dataset_dir, f'{dataset_name}.pkl'), 'rb') as f:
            X, y = pickle.load(f)

    ratio = sum(y==0) / sum(y==1) 
    print(f'this dataset imbalance ratio is:{ratio}') 
    return X, y


local_data = {
    'mfeat_fac':'/path/mfeat/mfeat-fac',
    'mfeat_pix':'/path/mfeat/mfeat-pix',
    'mfeat_kar':'/path/mfeat/mfeat-kar',
    # 'mfeat_zer':'/path/mfeat/mfeat-zer',
    'mfeat_fou':'/path/mfeat/mfeat-fou',
    # 'mfeat_mor':'/path/mfeat/mfeat-mor'
}

local_data_positive_label = {
    'mfeat_fac': 9,
    'mfeat_pix': 9,
    'mfeat_kar': 9,
    'mfeat_zer': 9,
    'mfeat_fou': 9,
    'mfeat_mor': 9,
}

def load_local_data(data, local_data_name):  # load downloaded file, for example mfeat, preprocess the file to create dataset
    local_file = data.get(local_data_name)
    with open(local_file) as f:
        lines = f.readlines()
        data = [line[:-1].split() for line in lines]
        X = np.array(data).astype(np.float64)

    X = (X - X.min())/(X.max() - X.min())  # scale to 0~1, not need for tree model
    y = np.repeat(range(10), 200)
    y = np.where(y == local_data_positive_label.get(local_data_name), 1, 0)

    return X, y


def plot_data(X, y, dataset_name=None):
    if X.shape[1]==1:
        print('feature number is less than 2...')
        return
    
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=10, cmap='rainbow')
    # plt.savefig(os.path.join(dataset_dir, f'./{dataset_name}.png'), dpi=300)
    plt.show()


if __name__ == '__main__':
    for dataset_name, _ in list(dataset_name_id.items()):
        print('\n\n'+'='*50 + f' load dataset:{dataset_name} ' + '='*50)
        load_UCI_data(dataset_name_id, dataset_name) # load data
        # plot_data(X, y, dataset_name)

    # for dataset_name, _ in list(synth_data.items()): 
    #     X, y = synthetic_data(synth_data, dataset_name)
    #     plot_data(X, y, dataset_name)

    # for dataset_name, _ in list(local_data.items()):
    #     X, y = load_local_data(dataset_name)
    #     print(X[:10])
    #     print(y[:10])