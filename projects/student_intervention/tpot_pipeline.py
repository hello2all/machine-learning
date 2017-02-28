import numpy as np

from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, Normalizer
from xgboost import XGBClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('student-data.csv', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    SelectPercentile(percentile=90, score_func=f_classif),
    Normalizer(norm="l1"),
    SelectKBest(k=1, score_func=f_classif),
    XGBClassifier(learning_rate=0.0001, max_depth=10, min_child_weight=3, n_estimators=500, subsample=0.79)
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
