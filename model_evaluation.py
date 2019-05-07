import numpy as np
import pandas as pd
import random
from sklearn.model_selection import RepeatedStratifiedKFold
from SessionSet import SessionSet
from BayesianModel import BayesianModel
from LogisticPrior import LogisticPrior

USER_ID_INDEX = 6

def compute_generalization_error_estimate(
	session_set,
	folds=3,
	repeats=5,
	stratification='user.country',
	top_x_cities=3,
	max_request_size_for_stats=2,
	ignore_coocurrences=False,
	aggregation=None,
	epsilon=1e-5,
	verbose=1,
	train_lr=False,
	check_history=False
):
	"""
	Function to compute the generalization error of the Bayesian Model through crossvalidation.
	It can be parametrized to aggregagate data, use a prior based on logistic regression
	and check for previous sessions of the same user. 
	"""
	splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats)
	model = BayesianModel(epsilon, ignore_coocurrences, aggregation)
	cities = session_set.get_list_of_cities()
	results = np.zeros((folds*repeats, top_x_cities, max_request_size_for_stats+1))
	train_results = np.zeros((folds*repeats, top_x_cities, max_request_size_for_stats+1))
	train_tracker = np.zeros((folds*repeats, 1, max_request_size_for_stats+1))
	test_tracker = np.zeros((folds*repeats, 1, max_request_size_for_stats+1))
	results_row_counter = 0
	lr_prior = None
	if verbose: print('Running model evaluation {0} folds x {1} repeats = {2} times'.format(folds,repeats,folds*repeats), end='')
	for train_index, test_index in splitter.split(session_set.df, session_set.df.loc[:, stratification]):
		if verbose: print('.', end='')
		train_set = session_set.df.iloc[train_index]
		test_set = session_set.df.iloc[test_index]
		model.fit(cities, train_set)
		if train_lr:
			lr_prior = LogisticPrior(session_set.df, ['month','day','join_month','join_day', 'centered_hour', 'country_index'], cities)
			lr_prior.fit(train_set)
		if verbose > 1: print('fold: {0}, repeat:{1}'.format(results_row_counter % folds, results_row_counter // folds))
		# Train error
		for train in train_set.values:
			targetIndex = random.randint(0, len(train[0])-1)
			target = train[0][targetIndex]
			query = [c for i,c in enumerate(train[0]) if i != targetIndex]
			query_length = min(len(query), max_request_size_for_stats)
			if train_lr:
				ranked_cities = model._predict(query, lr_prior.predict(train))
			else:
				ranked_cities = model._predict(query)
			train_tracker[results_row_counter, 0 , query_length] += 1
			for j in range(top_x_cities):
				if target in ranked_cities[:j+1]:
					train_results[results_row_counter, j, query_length] += 1
		# Test Error
		for test in test_set.values:
			targetIndex = random.randint(0, len(test[0])-1)
			target = test[0][targetIndex]
			query = [c for i,c in enumerate(test[0]) if i != targetIndex]
			query_length = min(len(query), max_request_size_for_stats)
			if check_history :
				history_query = set()
				[history_query.update(x) for x in train_set.loc[train_set['user.user_id'] == test[USER_ID_INDEX]]['cities'].values]
				history_query.update(query)
				query = list(history_query)
			if train_lr:
				ranked_cities = model._predict(query, lr_prior.predict(test))
			else:
				ranked_cities = model._predict(query)
			test_tracker[results_row_counter, 0 , query_length] += 1
			for j in range(top_x_cities):
				if target in ranked_cities[:j+1]:
					results[results_row_counter, j, query_length] += 1
		train_results[results_row_counter, :, :] /= len(train_index)
		train_tracker[results_row_counter, :, :] /= len(train_index)
		results[results_row_counter, :, :] /= len(test_index)
		test_tracker[results_row_counter, :, :] /= len(test_index)
		results_row_counter += 1
	if verbose: print('Done')
	return {
		'mean': np.mean(results/test_tracker, axis=0),
		'std': np.std(results/test_tracker, axis=0),
		'mean_train': np.mean(train_results/train_tracker, axis=0),
		'std_train': np.std(train_results/train_tracker, axis=0),
		'data': results,
		'tracker': test_tracker
	}
