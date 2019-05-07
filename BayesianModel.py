import pandas as pd
import numpy as np
from SessionSet import SessionSet

class BayesianModel:
	"""
	Main model uses Bayes Theorem to compute the likelihood of
	querying a city best on a request of size 0...n.
	Must be fit with sample sessions before callign predict.s
	"""

	def __init__(self, epsilon=1e-5, ignore_coocurrences=False, aggregation=None):
		self.epsilon = epsilon
		self.ignore_coocurrences = ignore_coocurrences
		self.aggregation = aggregation

	def _get_city_for_index(self, index):
		return self._i_c_m[index]

	def _predict(self, query_sequence, custom_prior=None):
		if self._fit_data is None:
			print('Please Fit Data First')
			return
		prior = custom_prior
		if prior is None:
			prior = np.diag(self._fit_data.values)
		if self.ignore_coocurrences:
			ll = prior
		else:
			ll = np.zeros((1,len(self._i_c_m)))
			for city in query_sequence:
				if city in self._c_i_m.keys():
					ll += self._fit_data.loc[city].values
			ll += max(len(query_sequence),1) * prior
		ll = ll * -1
		ordered_city = list(map(self._get_city_for_index, np.argsort(ll.flatten())))
		filtered_ordered_city = [city for city in ordered_city if city not in query_sequence]
		return filtered_ordered_city

	def _predict_multiple(self, list_of_sequences, custom_priors=[]):
		if self._fit_data is None:
			print('Please Fit Data First')
			return
		output = []
		for i, sequence in enumerate(list_of_sequences):
			if i < len(custom_priors):
				output.append(self._predict(sequence, priors[i]))
			else:
				output.append(self._predict(sequence))
		return output


	def predict(self, input_to_predict, custom_prior=None):
		if self._fit_data is None:
			print('Please Fit Data First')
			return
		if isinstance(input_to_predict, list) and \
			any(isinstance(el, list) for el in input_to_predict):
			return self._predict_multiple(input_to_predict, custom_prior)
		elif isinstance(input_to_predict, list) and \
			any(isinstance(el, str) for el in input_to_predict):
			return self._predict(input_to_predict, custom_prior)

	def compute_logproba_cooccurence_matrix(self, cities, df):
		#NOTE: assumes the same city is not queried twice in a session
		train_df = df
		if self.aggregation:
			aggregate = ['cities'] + self.aggregation
			train_df = df.loc[:, aggregate].groupby(self.aggregation).agg('sum')
		city_count = len(cities)
		session_count = len(train_df.index)
		city_index_mapping = {city: index for index, city in enumerate(cities)}
		index_city_mapping = {index: city for index, city in enumerate(cities)}
		X_total_counts = np.zeros((1,city_count), dtype=np.single)
		X_cooccurence_counts = np.zeros((city_count, city_count), dtype=np.single)


		for session_cities in train_df['cities']:
			cities_unique = list(set(session_cities))
			for i in range(len(cities_unique)):
				X_total_counts[0,city_index_mapping[cities_unique[i]]] += 1.0
				for j in range(i,len(cities_unique)):
					X_cooccurence_counts[city_index_mapping[cities_unique[i]],city_index_mapping[cities_unique[j]]] += 1.0
		X_total_counts[X_total_counts == 0] = 1 # avoid dividing by 0
		X_cooccurence_probs = np.log(X_cooccurence_counts / X_total_counts) 
		for i in range(city_count):
			X_cooccurence_probs[i,i] = np.log(X_total_counts[0,i] / session_count)

		data = pd.DataFrame(X_cooccurence_probs, index=cities, columns=cities)
		data.fillna(np.log(self.epsilon), inplace=True)
		data.replace(-np.inf,np.log(self.epsilon), inplace=True)
		return data, city_index_mapping, index_city_mapping

	def fit(self, cities, df):
		self._fit_data, self._c_i_m, self._i_c_m  = self.compute_logproba_cooccurence_matrix(cities, df)
