from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier, EasyEnsembleClassifier, RUSBoostClassifier
from sklearn.tree import DecisionTreeClassifier


BASE_ESTIMATOR = DecisionTreeClassifier(min_samples_split=10)
ESTIMATOR = AdaBoostClassifier(n_estimators=10, estimator=BASE_ESTIMATOR)
RF  = RandomForestClassifier(n_estimators=10, min_samples_split=10)

model_dict = {
    "BB": BalancedBaggingClassifier(n_estimators=100, estimator=BASE_ESTIMATOR),
    "BRF": BalancedRandomForestClassifier(n_estimators=100, min_samples_split=10),
    "RUSBoost": RUSBoostClassifier(n_estimators=100, estimator=BASE_ESTIMATOR),
    "EE":EasyEnsembleClassifier(n_estimators=10, estimator=ESTIMATOR, max_features=1.0),
    # modify EasyEnsembleClassifier to add DWUS
    "BRFSE_0.1":EasyEnsembleClassifier(n_estimators=10, estimator=ESTIMATOR, max_features=0.1, sampler_type="DWUS"),
    "BRFSE_0.2":EasyEnsembleClassifier(n_estimators=10, estimator=ESTIMATOR, max_features=0.2, sampler_type="DWUS"),
    "BRFSE_0.3":EasyEnsembleClassifier(n_estimators=10, estimator=ESTIMATOR, max_features=0.3, sampler_type="DWUS"),
    "BRFSE_0.4":EasyEnsembleClassifier(n_estimators=10, estimator=ESTIMATOR, max_features=0.4, sampler_type='DWUS'),
    "BRFSE_0.5":EasyEnsembleClassifier(n_estimators=10, estimator=ESTIMATOR, max_features=0.5, sampler_type='DWUS'),
    "BRFSE_0.6":EasyEnsembleClassifier(n_estimators=10, estimator=ESTIMATOR, max_features=0.6, sampler_type='DWUS'),
    "BRFSE_0.7":EasyEnsembleClassifier(n_estimators=10, estimator=ESTIMATOR, max_features=0.7, sampler_type='DWUS'),
    "BRFSE_0.8":EasyEnsembleClassifier(n_estimators=10, estimator=ESTIMATOR, max_features=0.8, sampler_type='DWUS'),
    "BRFSE_0.9":EasyEnsembleClassifier(n_estimators=10, estimator=ESTIMATOR, max_features=0.9, sampler_type='DWUS'),
    "BRFSE_1.0":EasyEnsembleClassifier(n_estimators=10, estimator=ESTIMATOR, max_features=1.0, sampler_type='DWUS'),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, estimator=BASE_ESTIMATOR),
    "RF":RandomForestClassifier(n_estimators=100, min_samples_split=10),
}