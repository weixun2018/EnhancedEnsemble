import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import  f1_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
from dataloader import dataset_name_id, load_UCI_data
from dataloader import synthetic_data, synth_data
from dataloader import local_data, load_local_data
from model import model_dict
from sampler import get_sampler, sampler_dict
from sklearn.model_selection import ShuffleSplit, KFold
import time

DATA_TYPE_DICT_FUN = {
    "UCI_python":(dataset_name_id, load_UCI_data),  # data from UCI, can import by python
    "UCI_local":(local_data, load_local_data),  # data from UCI, download to local
    "synth":(synth_data, synthetic_data)   # synthetic data by sklearn.datasets.make_classification
    }

DATA_TYPE = 'UCI_python'
MODEL_ZOO =  model_dict

def print_performance(pred, score, y_test, model_name, sampler_name=None):
    f1_value = f1_score(y_test, pred)
    AUC = roc_auc_score(y_test, score)
    g_mean = geometric_mean_score(y_test, pred)
    return AUC, f1_value, g_mean

AUC_record = defaultdict(dict)
F1_record = defaultdict(dict)
GMEAN_record = defaultdict(dict)

start = time.time()

# loop for dataset, must in the out loop (keep the same input data for varied model)
DATA = DATA_TYPE_DICT_FUN.get(DATA_TYPE)[0]
for dataset_name, _ in list(DATA.items()):
    AUC_record[dataset_name] = defaultdict(list)
    F1_record[dataset_name] = defaultdict(list)
    GMEAN_record[dataset_name] = defaultdict(list)
    print('\n\n'+'='*50 + f' load dataset:{dataset_name} ' + '='*50)  
    X, y = DATA_TYPE_DICT_FUN.get(DATA_TYPE)[1](DATA, dataset_name) # load data
    for i in range(10):
        kf = KFold(n_splits=5, shuffle=True)
        for train, test in kf.split(X):
            x_train, x_test, y_train, y_test = X[train], X[test], y[train], y[test]
            for sampler_name, _ in sampler_dict.items():
                sampler = get_sampler(sampler_name)
                if sampler is not None:
                    x_train_resample, y_train_resample = sampler.fit_resample(x_train, y_train)
                else:
                    x_train_resample, y_train_resample = x_train, y_train

                for model_name, _ in MODEL_ZOO.items():
                    if sampler is not None: # here only use RUS
                        model_sampler = f'{sampler_name}+{model_name}'
                        if model_name not in ['AdaBoost', 'RF']:
                            continue
                    else:
                        model_sampler = f'{model_name}'

                    print('\n' + '*'*30 + f' get model:{model_sampler} ' + '*'*30)

                    model = MODEL_ZOO[model_name]
                    model.fit(x_train_resample, y_train_resample)
                    y_pred = model.predict(x_test)
                    y_score = model.predict_proba(x_test)[:, 1]  # fix bug that use y_pred to calc AUC!
                    AUC, f1, g_mean = print_performance(y_pred, y_score, y_test, model_name, sampler_name)
                    AUC_record[dataset_name][model_sampler].append(AUC)
                    F1_record[dataset_name][model_sampler].append(f1)
                    GMEAN_record[dataset_name][model_sampler].append(g_mean)

print(f'=====> use time:{(time.time()-start)//60} minutes')

AUC_df = pd.DataFrame(AUC_record).map(lambda x: np.array(x))
F1_df = pd.DataFrame(F1_record).map(lambda x: np.array(x))
GMEAN_df = pd.DataFrame(GMEAN_record).map(lambda x: np.array(x))


AUC_MEAN_STD_df = AUC_df.map(lambda x:(f'{x.mean():.3f}|{x.std():.3f}')) # with deviation
# AUC_MEAN_STD_df = AUC_df.applymap(lambda x:(f'{x.mean():.3f}')) # without deviation
AUC_MEAN_STD_df['Average'] = AUC_df.map(lambda x:x.mean()).mean(axis=1)
F1_MEAN_STD_df = F1_df.map(lambda x:(f'{x.mean():.3f}|{x.std():.3f}')) # with deviation
F1_MEAN_STD_df['Average'] = F1_df.map(lambda x:x.mean()).mean(axis=1)
GMEAN_MEAN_STD_df = GMEAN_df.map(lambda x:(f'{x.mean():.3f}|{x.std():.3f}')) # with deviation
GMEAN_MEAN_STD_df['Average'] = GMEAN_df.map(lambda x:x.mean()).mean(axis=1)

print(AUC_MEAN_STD_df)
print(F1_MEAN_STD_df)
print(GMEAN_MEAN_STD_df)

AUC_df.to_csv('result/UCI_python/auc_origin_8.csv')
F1_df.to_csv('result/UCI_python/f1_origin_8.csv')
GMEAN_df.to_csv('result/UCI_python/gmean_origin_8.csv')

AUC_MEAN_STD_df.to_csv('result/UCI_python/auc_mean_std_DWUS_8.csv')
F1_MEAN_STD_df.to_csv('result/UCI_python/f1_mean_std_DWUS_8.csv')
GMEAN_MEAN_STD_df.to_csv('result/UCI_python/gmean_mean_std_DWUS_8.csv')