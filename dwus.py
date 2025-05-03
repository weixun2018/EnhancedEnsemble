import numpy as np
from sklearn.utils import _safe_indexing, check_random_state

from imblearn.utils import Substitution, check_target_type
from imblearn.utils._docstring import _random_state_docstring
from imblearn.utils._validation import _check_X, _count_class_sample
from imblearn.under_sampling.base import BaseUnderSampler
from sklearn.metrics import euclidean_distances

def calculate_sample_mini_distance(sample, group, metric='euclidean'):
    if metric == 'euclidean':
        return min(np.sqrt(np.sum((sample - group)**2, axis=1)))  # faster computing
    elif metric == 'cosine':
        return min(np.dot(sample,group.T)/(np.linalg.norm(sample)*np.linalg.norm(group)))
    else:
        raise ValueError('not supported metric!')

class DWUS(BaseUnderSampler):
    """Class to perform Distance Weighted Under-Sampling.

    Under-sample the majority class(es) by  picking samples on the probability
    based on its distance to minority class centroid.

    Read more in the :ref:`User Guide <controlled_under_sampling>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    replacement : bool, default=False
        Whether the sample is with or without replacement.

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    sample_indices_ : ndarray of shape (n_new_samples,)
        Indices of the samples selected.

        .. versionadded:: 0.4

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    NearMiss : Undersample using near-miss samples.

    Notes
    -----
    Supports multi-class resampling by sampling each class independently.
    Supports heterogeneous data as object array containing string and numeric
    data.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import DWUS
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ...  weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> dwus = DWUS(random_state=42)
    >>> X_res, y_res = dwus.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 100, 1: 100}})
    """

    _parameter_constraints: dict = {
        **BaseUnderSampler._parameter_constraints,
        "replacement": ["boolean"],
        "random_state": ["random_state"],
    }

    def __init__(
        self, *, sampling_strategy="auto", random_state=None, replacement=False, metric='euclidean'
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.replacement = replacement
        self.metric = metric

    def _check_X_y(self, X, y):
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X = _check_X(X)
        self._check_n_features(X, reset=True)
        self._check_feature_names(X, reset=True)
        return X, y, binarize_y
    
    def _get_minority_sample_centroid(self, X, y):
        target_stats = _count_class_sample(y)
        class_minority = min(target_stats, key=target_stats.get)
        assert class_minority == 1
        minority_class_indices = np.flatnonzero(y == class_minority)
        minority_class_sample = _safe_indexing(X, minority_class_indices)
        minority_class_centroid = np.average(minority_class_sample, axis=0)
        return minority_class_centroid
    
    def _get_class_sample(self, X, y, sample_class):
        sample_class_indices = np.flatnonzero(y == sample_class)
        sample_class_sample = _safe_indexing(X, sample_class_indices)
        return sample_class_sample
    
    def _get_class_sample_centroid(self, X, y, sample_class):
        sample_class_indices = np.flatnonzero(y == sample_class)
        sample_class_sample = _safe_indexing(X, sample_class_indices)
        sample_class_centroid = np.average(sample_class_sample, axis=0)
        return sample_class_centroid

    def _fit_resample(self, X, y):

        idx_under = np.empty((0,), dtype=int)

        minority_class_samples = self._get_class_sample(X, y, 1)

        for target_class in np.unique(y):
            if target_class in self.sampling_strategy_.keys():
                n_samples = self.sampling_strategy_[target_class]
                target_class_index = np.flatnonzero(y == target_class)
                target_class_sample = np.array(_safe_indexing(X, target_class_index))
                target_sample_distance = [calculate_sample_mini_distance(np.reshape(b, (1,-1)), minority_class_samples, metric=self.metric) for b in target_class_sample]
                target_sample_distance = np.reshape(target_sample_distance, (-1,))
                target_sample_distance_inverse = [1/(x+0.001) for x in target_sample_distance]  # sample more for nearer samples
                sample_probability = target_sample_distance_inverse / np.sum(target_sample_distance_inverse)
                index_target_class = sorted(np.random.choice(range(len(target_class_sample)), n_samples, replace=False, p=sample_probability))  # sample without replacement
            else:
                index_target_class = slice(None)

            idx_under = np.concatenate(
                (
                    idx_under,
                    np.flatnonzero(y == target_class)[index_target_class],
                ),
                axis=0,
            )

        self.sample_indices_ = idx_under

        return _safe_indexing(X, idx_under), _safe_indexing(y, idx_under)

    def _more_tags(self):
        return {
            "X_types": ["2darray", "string", "sparse", "dataframe"],
            "sample_indices": True,
            "allow_nan": True,
            "_xfail_checks": {
                "check_complex_data": "Robust to this type of data.",
            },
        }
