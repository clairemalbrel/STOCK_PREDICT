import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
<<<<<<< HEAD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
=======
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline
from tpot.export_utils import set_param_recursive
>>>>>>> a7a1995d6cace0bc06b41649b6c8333080b15e32

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
<<<<<<< HEAD
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.8381355643981907
exported_pipeline = make_pipeline(
    RFE(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.15000000000000002, n_estimators=100), step=0.9500000000000001),
    RFE(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.1, n_estimators=100), step=0.6500000000000001),
    StackingEstimator(estimator=SGDClassifier(alpha=0.001, eta0=1.0, fit_intercept=False, l1_ratio=0.5, learning_rate="constant", loss="log", penalty="elasticnet", power_t=1.0)),
    PCA(iterated_power=3, svd_solver="randomized"),
    StackingEstimator(estimator=SGDClassifier(alpha=0.001, eta0=0.1, fit_intercept=True, l1_ratio=0.5, learning_rate="constant", loss="hinge", penalty="elasticnet", power_t=0.5)),
    RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.2, min_samples_leaf=19, min_samples_split=3, n_estimators=100)
)
=======
            train_test_split(features, tpot_data['target'], random_state=32)

# Average CV score on the training set was: 0.8027429467084639
exported_pipeline = make_pipeline(
    PCA(iterated_power=10, svd_solver="randomized"),
    BernoulliNB(alpha=1.0, fit_prior=True)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 32)
>>>>>>> a7a1995d6cace0bc06b41649b6c8333080b15e32

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
