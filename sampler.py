from imblearn.under_sampling import RandomUnderSampler, NearMiss
# from Sample.PRUS import PRUS
from imblearn.over_sampling import SMOTE, RandomOverSampler

sampler_dict = {
    "NoSample":None,
    "RUS": RandomUnderSampler(),
    # "ROS": RandomOverSampler(),
    # "SMOTE": SMOTE(),
    # "PRUS_euc": PRUS(metric='euclidean'),
    # "PRUS_cos":PRUS(metric='cosine'),
    # "NearMiss": NearMiss(),
}

def get_sampler(sampler_name=None):
    if sampler_name:
        return sampler_dict[sampler_name] 