import pandas as pd
import numpy as np
csv_file_1 = 'result/UCI_python/f1_mean_std_PRUS_8.csv'
csv_file_2 = 'result/UCI_local/f1_mean_std_PRUS_4.csv'


df_1 = pd.read_csv(csv_file_1, index_col='Unnamed: 0')
df_2 = pd.read_csv(csv_file_2, index_col='Unnamed: 0')
df1 = df_1.drop('Average', axis=1)
df2 = df_2.drop('Average', axis=1)
print(df1.index)

df1 = df1.reindex(columns=sorted(df1.columns))

df = pd.concat([df1, df2], axis=1)

df = df.reindex(index=['AdaBoost', 'RF', 'RUS+AdaBoost', 'RUS+RF','BB', 'BRF', 'RUSBoost', 'EE', 
                  'BRFSE_0.1', 'BRFSE_0.2', 'BRFSE_0.3','BRFSE_0.4', 'BRFSE_0.5', 'BRFSE_0.6', 
                  'BRFSE_0.7', 'BRFSE_0.8','BRFSE_0.9', 'BRFSE_1.0'])

df.loc['BRFSE_best'] = df.iloc[8:].apply(max, axis=0)

df_ = df.map(lambda x: float(str(x).split('|')[0]))
df['Average'] = df_.mean(axis=1)

print(df)

df.to_csv('result/f1_mean_std_PRUS_final_12.csv')