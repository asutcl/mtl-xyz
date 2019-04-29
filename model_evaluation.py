import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from SessionSet import SessionSet

def compute_generalization_error_estimate(session_set, folds=3, repeats=5, stratification='user.country'):
	splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats)
	for train_index, test_index in splitter.split(session_set.df.loc[:,'cities'],session_set.df.loc[:,'user.country']):
		train_set = SessionSet(session_set.df.iloc[train_index], session_set.country)
		test_set = SessionSet(session_set.df.iloc[test_index], session_set.country)
		print(train_set.session_count, test_set.session_count)