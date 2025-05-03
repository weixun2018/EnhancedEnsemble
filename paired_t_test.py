import pandas as pd
import numpy as np
import scipy.stats as stats
csv_file_1 = 'result/UCI_python/gmean_origin_8.csv'
csv_file_2 = 'result/UCI_local/gmean_origin_4.csv'

df1 = pd.read_csv(csv_file_1, index_col='Unnamed: 0')
df2 = pd.read_csv(csv_file_2, index_col='Unnamed: 0')

df1 = df1.reindex(columns=sorted(df1.columns))

df = pd.concat([df1, df2], axis=1)
print(df.columns)

df = df.reindex(index=['AdaBoost', 'RF', 'RUS+AdaBoost', 'RUS+RF','BB', 'BRF', 'RUSBoost', 'EE', 
                  'BRFSE_0.1', 'BRFSE_0.2', 'BRFSE_0.3','BRFSE_0.4', 'BRFSE_0.5', 'BRFSE_0.6', 
                  'BRFSE_0.7', 'BRFSE_0.8','BRFSE_0.9', 'BRFSE_1.0'])
print(df.index)

df = df.map(lambda x: np.array([float(y) for y in x[1:-1].split()]))
MEAN_STD_df = df.map(lambda x:x.mean())

df.loc['BRFSE_best'] = np.array(df.iloc[8:]).T[(MEAN_STD_df.iloc[8:] == MEAN_STD_df.iloc[8:].max(axis=0)).T]

arr = np.array(df)

samples = np.vstack((arr[:8], arr[-1:]))
m, n = samples.shape 
print(samples.shape)
# print(samples[:2,:2])
methods = ['Ada', 'RF', 'UnderAda', 'UnderRF', 'BB', 'BRF', 'RUSB', 'Easy', 'Enhanced']

method_paired_df = pd.DataFrame(index=methods, columns=methods)
for i in method_paired_df.index:
    for c in method_paired_df.columns:
        method_paired_df.loc[i, c] = [0, 0, 0]
        
# print(method_paired_df)

for i in range(m):
    for j in range(i+1, m):
        for k in range(n):
            statics, p_value = stats.ttest_rel(arr[i][k], arr[j][k])
            if p_value < 0.05:
                if samples[i][k].mean() > samples[j][k].mean():
                    method_paired_df.iloc[i, j][0] += 1
                    method_paired_df.iloc[j, i][2] += 1
                else:
                    method_paired_df.iloc[i, j][2] += 1
                    method_paired_df.iloc[j, i][0] += 1
            else:
                method_paired_df.iloc[i, j][1] += 1
                method_paired_df.iloc[j, i][1] += 1

print(method_paired_df)
method_paired_df = method_paired_df.map(lambda x: '& '+ '-'.join(str(y) for y in x))
print(method_paired_df)
