import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification

def calculate_sample_mini_distance(sample, group, metric='euclidean'):
    if metric == 'euclidean':
        return min(np.sqrt(np.sum((sample - group)**2, axis=1)))  # faster computing
    elif metric == 'cosine':
        return min(np.dot(sample,group.T)/(np.linalg.norm(sample)*np.linalg.norm(group)))
    else:
        raise ValueError('not supported metric!')

X, y = make_classification(n_samples=100, n_features=2, n_classes=2, weights=[0.9,0.1],n_redundant=0, random_state=0)
minority_class_samples = X[y==1]
majority_class_samples = X[y==0]

plt.figure(figsize=(6, 6), dpi=300)
plt.scatter(x=minority_class_samples[:,0],y=minority_class_samples[:,1], color='red', label='Minority Class Samples')
colors = [calculate_sample_mini_distance(x, minority_class_samples) for x in majority_class_samples]
colors = [1/(x+0.001) for x in colors]
colors = colors / max(colors)

cmap = plt.cm.Blues
norm = plt.Normalize(vmin=0, vmax=1)

plt.scatter(x=majority_class_samples[:,0],y=majority_class_samples[:,1], c=colors, cmap=cmap, norm=norm, label='Majority Class Samples')
plt.colorbar()

plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig('result/figure/dwus-examples.png')