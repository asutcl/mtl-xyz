import numpy as np
import pandas as pd
import random
from sklearn.model_selection import RepeatedStratifiedKFold
from SessionSet import SessionSet
from BayesianModel import BayesianModel

def compute_generalization_error_estimate(
	session_set,
	folds=3,
	repeats=5,
	stratification='user.country',
	top_x_cities=3,
	max_request_size_for_stats=2,
	ignore_coocurrences=False,
	aggregation=None,
	epsilon=1e-5
):
	splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats)
	model = BayesianModel(epsilon, ignore_coocurrences, aggregation)
	cities = session_set.get_list_of_cities()
	results = np.zeros((folds*repeats, top_x_cities, max_request_size_for_stats+1))
	test_tracker = np.zeros((folds*repeats, 1, max_request_size_for_stats+1))
	results_row_counter = 0
	for train_index, test_index in splitter.split(session_set.df.loc[:,'cities'],session_set.df.loc[:,'user.country']):
		train_set = session_set.df.iloc[train_index]
		test_set = session_set.df.iloc[test_index]
		model.fit(cities, train_set)
		print('fold: {0}, repeat:{1}'.format(results_row_counter % folds, results_row_counter // folds))
		for test in test_set.values:
			targetIndex = random.randint(0, len(test[0])-1)
			target = test[0][targetIndex]
			query = [c for i,c in enumerate(test[0]) if i != targetIndex]
			query_length = min(len(query), max_request_size_for_stats)
			ranked_cities = model._predict(query)
			test_tracker[results_row_counter, 0 , query_length] += 1
			for j in range(top_x_cities):
				if target in ranked_cities[:j+1]:
					results[results_row_counter, j, query_length] += 1
		results[results_row_counter, :, :] /= len(test_index)
		test_tracker[results_row_counter, :, :] /= len(test_index)
		results_row_counter += 1
	return {
		'mean': np.mean(results/test_tracker, axis=0),
		'std': np.std(results/test_tracker, axis=0),
		'data': results,
		'tracker': test_tracker
	}
