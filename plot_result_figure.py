import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
               
csv_file_list = ['result/auc_mean_std_PRUS_final_12.csv',
                 'result/gmean_mean_std_PRUS_final_12.csv',
                 'result/f1_mean_std_PRUS_final_12.csv',
                ]
title_list = ['(a) AUC of models on 12 UCI datasets',
              '(b) G-mean of models on 12 UCI datasets',
              '(c) F-measure of models on 12 UCI datasets',
             ]
value_type = ['AUC', 'G-mean', 'F1']

matplotlib.rcParams['font.family'] = 'Arial'

plt.figure(figsize=(24, 50), dpi=300)

for i in range(len(csv_file_list)):
    plt.subplot(3, 1, i+1)
    df = pd.read_csv(csv_file_list[i], index_col='Unnamed: 0')
    df = df.map(lambda x: float(str(x).split('|')[0]))
    # print(df)
    res = df.loc[['EE', 'BRFSE_1.0', 'BRFSE_best']]
    labels = res.columns
    x = range(len(labels))
    plt.tick_params(axis='y', labelsize=36)
    plt.bar(x, res.loc['EE'], width=0.2, label='Baseline')
    plt.bar([xi+0.2 for xi in x], res.loc['BRFSE_1.0'], width=0.2, label='w/ DWUS, w/o RFS')
    plt.bar([xi+0.4 for xi in x], res.loc['BRFSE_best'], width=0.2, label='w/ DWUS & RFS')
    plt.xlabel('Dataset', fontsize=40)
    plt.ylabel(value_type[i], fontsize=40)
    plt.title(title_list[i], y=-0.25, font={'family':'Times New Roman', 'size':60})
    plt.xticks([p + 0.2 for p in x], labels, rotation=30, ha='right', fontsize=36)
    y_start = 0 if value_type[i]=='F1' else 0.4
    plt.ylim(y_start, 1.05)

plt.legend(fontsize=28)
plt.tight_layout(pad=5)
# plt.show()
plt.savefig('result/figure_final/DWUS_and_RFS.png')

title_list = ['(a) AUC of method over feature sampling rate on 12 UCI datasets',
              '(b) G-mean of method over feature sampling rate on 12 UCI datasets',
              '(c) F-measure of method over feature sampling rate on 12 UCI datasets',
             ]

# different label for sub-figures, use axes
fig, axes = plt.subplots(3, 2, figsize=(24,40))
columns_split = [['abalone', 'bank', 'vehicle', 'yeast','wine','mfeat_fou'], ['isolet','musk2','wave','mfeat_kar', 'mfeat_fac', 'mfeat_pix']]
fig.subplots_adjust(
    # left=0.1, right=0.9, bottom=0.1, top=0.9,  
    hspace=0.3 
)
for i in range(len(csv_file_list)):
    df = pd.read_csv(csv_file_list[i], index_col='Unnamed: 0')
    df = df.map(lambda x: float(str(x).split('|')[0]))
    result = df.iloc[8:-1]
    # print(df)
    print(result.columns)
    labels = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    markers = '^os*d.'
    x = range(len(labels))
    for k in range(len(columns_split)):
        res = result[columns_split[k]]
        # print(type(axes[i]))
        for j in range(len(res.columns)):
            lab = res.columns[(j-1)%len(res.columns)] if 'Average' in res.columns else res.columns[j] # average first!
            line_style = '-' if lab=='Average' else '-.'
            line_width = 2 if lab=='Average' else 1
            axes[i][k].plot(labels, res[lab],label=lab, linestyle=line_style, linewidth= line_width, marker=markers[j%6], markersize=8, markeredgecolor='red', markerfacecolor='yellow')
        axes[i][k].set_xlabel(r'Feature sampling rate($\alpha$)', fontsize=32)
        axes[i][k].set_ylabel(value_type[i], fontsize=32)
        if k == 0:
            axes[i][k].set_title(title_list[i], x=1.1, y=-0.15, font={'family':'Times New Roman', 'size':40})
        axes[i][k].set_xticks(labels, labels, fontsize=30)
        axes[i][k].tick_params(axis='y', labelsize=30)
        axes[i][k].legend(fontsize=20)
    
plt.tight_layout(pad=5)
# plt.show()
plt.savefig('result/figure_final/alpha-sensitivity.png')