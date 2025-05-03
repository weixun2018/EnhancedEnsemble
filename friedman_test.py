import scipy.stats as stats
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

csv_file_list = ['result/auc_mean_std_DWUS_final_12.csv',
                 'result/gmean_mean_std_DWUS_final_12.csv',
                 'result/f1_mean_std_DWUS_final_12.csv',
                ]

def draw_cd_graph(title):
    # base setting
    ax = plt.gca()
    ax.axis('off')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim((4-n)*10, 140)
    plt.title(title, y=-0.1, font={'family':'Times New Roman', 'size':20})
    
    tics = np.linspace(0, 1, k)
    plt.plot([tics[0], tics[-1]], [100, 100], color='black', lw=2)
    for i in range(k):
        plt.plot([tics[i], tics[i]], [100, 103], color='black', lw=2)
        plt.text(tics[i], 105, f"{k-i}", ha='center', va='bottom', fontsize=12)
    
    # draw CD line
    plt.plot([0, 0, 0, cd/(k-1), cd/(k-1), cd/(k-1)], [124, 120, 122, 122, 120, 124], color='r', lw=1)
    plt.text(cd/(k-1)/2.2, 126, f"CD", color='r', fontsize=12)
    
    # draw method labels
    for i in range(k):
        x = tics[k-1 - i]  # inverse ranks
        if i <= k//2:
            plt.plot([(k-sorted_ranks[i])/(k-1), (k-sorted_ranks[i])/(k-1), 1], [100, 100-10*(n+1)-10*i, 100-10*(n+1)-10*i], color='b')
            plt.text(1.02, 100 - 10*(n+1) - 10*i, sorted_labels[i], 
                     ha='left', va='center', fontsize=14, color='black')
        else:
            plt.plot([(k-sorted_ranks[i])/(k-1), (k-sorted_ranks[i])/(k-1), 0], [100, 100-10*(n+1)-10*(k-i-1), 100-10*(n+1)-10*(k-i-1)], color='b')
            plt.text(-0.02, 100-10*(n+1)-10*(k-i-1), sorted_labels[i], 
                     ha='right', va='center', fontsize=14, color='black')
    
    # draw groups of methods that are not significantly different
    y_offset = 90
    for i in range(len(clique)):
        if not np.any(clique[i]):
            continue
        start = max(i, np.min(np.where(clique[i])[0]))
        end = np.max(np.where(clique[i])[0])
        x1 = (k - sorted_ranks[start]+0.1)/(k-1)
        x2 = (k - sorted_ranks[end]-0.1)/(k-1)
        plt.plot([x1, x2], [y_offset, y_offset], color='g', lw=2)
        y_offset -= 10

plt.figure(figsize=(8, 12), dpi=300)
title_list = ['(a) Average ranks of all methods on AUC',
              '(b) Average ranks of all methods on G-mean',
              '(c) Average ranks of all methods on F-measure',
             ]
for i in range(len(csv_file_list)):
    plt.subplot(3, 1, i+1)
    df = pd.read_csv(csv_file_list[i], index_col='Unnamed: 0')
    # print(df.columns)
    df = df.drop('Average', axis=1)
    df = df.map(lambda x: float(str(x).split('|')[0]))
    arr = np.array(df)
    samples = np.vstack((arr[:8], arr[-1:]))
    k, N = samples.shape  # k: methods number，N: datasets number
    labels = ['Ada', 'RF', 'UnderAda', 'UnderRF', 'UB', 'BRF', 'RUSB', 'EasyEn', 'Enhanced']
    
    (statistic, p_value), rank_ = stats.friedmanchisquare(*samples)

    t_f = (N-1) * statistic / (N * (k-1) - statistic)
    if t_f > 2.05:
        print("reject the hypothesis that all algorithms are equivalent at a significance level 0.05")
    else:
        print("agree the hypothesis that all algorithms are equivalent at a significance level 0.05")

    cd = 3.1 * math.sqrt(k*(k+1)/(6*N))  # 3.1 find in tables
    print(t_f, cd)
    rank = 10 - rank_
    print(rank)

    # sort ranks
    sorted_idx = np.argsort(rank)
    sorted_ranks = rank[sorted_idx]
    sorted_labels = np.array(labels)[sorted_idx]
    
    # find groups of methods that are not significantly different
    clique = np.abs(sorted_ranks[:, None] - sorted_ranks[None, :]) < cd
    temp = clique[:]
    for j in range(k-1, 0, -1):
        if np.all(temp[j-1, temp[j]] == temp[j, temp[j]]):
            temp[j] = False   
    n = temp.sum(axis=1)  
    temp = temp[n > 1]   
    n = len(temp)   

    draw_cd_graph(title_list[i])

plt.tight_layout()
plt.savefig('result/figure_final/friedman-test.png')

    