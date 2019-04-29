import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import helpers

class SessionSet:
	def __init__(self, dataframe, country, epsilon=1e-5):
		self.df = dataframe
		self.country = country
		self.index = helpers.get_index_for_country(country)
		self.color = helpers.get_color_for_country(country)
		self.session_count = self.df.shape[0]
		self.unique_users = set(self.df['user.user_id'].unique())
		self.epsilon = epsilon

	def get_time_metrics(self):
		time_by_hours = self.df.loc[:,'time_of_day']//helpers.SECONDS_PER_HOUR
		return pd.Series(time_by_hours.describe())

	def add_time_scatter_to_plot(self, ax):
		scatter_visualisation_y_offset = 0.5
		ys = np.random.random_sample(((self.session_count,),)) - scatter_visualisation_y_offset + self.index
		ax.scatter(ys, self.df.loc[:,'time_of_day']/helpers.SECONDS_PER_HOUR_F, c=self.color)

	def get_requests_grouped_by_user(self):
		sdf = self.df[['cities','user.user_id']]
		users_grouped = sdf.groupby(['user.user_id']).agg(lambda requests: requests.tolist())
		users_grouped['session_count'] = users_grouped.apply(lambda row: len(row['cities']), axis=1)
		users_grouped['cities_flat'] = users_grouped.apply(lambda row: helpers.flatten_list_of_lists(row['cities']), axis=1)
		users_grouped['cities_count'] = users_grouped.apply(lambda row: len(row['cities_flat']), axis=1)
		return users_grouped

	def get_list_of_cities(self):
		set_of_cities = set()
		for cities in self.df['cities']:
			set_of_cities.update(cities)
		return list(set_of_cities)

	def get_cities_occurence_count(self):
		city_occurences = {}
		for cities in self.df['cities']:
			for city in cities:
				city_occurences[city] = city_occurences.get(city,0) + 1
		return city_occurences

	def compute_logproba_cooccurence_matrix(self, epsilon):
		#NOTE assumes the same city is not queried twice in a session
		cities = self.get_list_of_cities()
		city_count = len(cities)
		city_index_mapping = {city: index for index, city in enumerate(cities)}
		index_city_mapping = {index: city for index, city in enumerate(cities)}
		X_total_counts = np.zeros((1,city_count), dtype=np.single)
		X_cooccurence_counts = np.zeros((city_count, city_count), dtype=np.single)
		for session_cities in self.df['cities']:
			for i in range(len(session_cities)):
				X_total_counts[0,city_index_mapping[session_cities[i]]] += 1.0
				for j in range(i,len(session_cities)):
					X_cooccurence_counts[city_index_mapping[session_cities[i]],city_index_mapping[session_cities[j]]] += 1.0
		X_cooccurence_probs = np.log(X_cooccurence_counts / X_total_counts)
		for i in range(city_count):
			X_cooccurence_probs[i,i] = np.log(X_total_counts[0,i] / self.session_count)
		data = pd.DataFrame(X_cooccurence_probs, index=cities, columns=cities)
		data.fillna(np.log(epsilon), inplace=True)
		data.replace(-np.inf,np.log(epsilon), inplace=True)
		return data, city_index_mapping, index_city_mapping

	def bayes_fit(self):
		self._fit_data, self._c_i_m, self._i_c_m  = compute_logproba_cooccurence_matrix(self.epsilon)

	def _get_city_for_index(self, index):
		return self._i_c_m[index]

	def _predict(self, query_sequence, custom_prior=None):
		if self._fit_data is None:
			print('Please Fit Data First')
			return
		prior = custom_prior
		if prior is None:
			prior = np.diag(self._fit_data.values)
		ll = np.zeros((1,len(self._i_c_m)))
		for city in query_sequence:
		    ll = ll + log_proba.loc[city].values
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
			if i < len(priors):
				output.append(self._predict(sequence, priors[i]))
			else:
				output.append(self._predict(sequence))
		return output

	def predict(self, input_to_predict, custom_priors=[]):
		if self._fit_data is None:
			print('Please Fit Data First')
			return
		if isinstance(input_to_predict, list) and \
			any(isinstance(el, list) for el in input_to_predict):
			return self._predict_multiple(input_to_predict)
		elif isinstance(input_to_predict, list) and \
			any(isinstance(el, str) for el in input_to_predict):
			return self._predict(input_to_predict)

	def generate_prior(self):


	def generate_test_query_data(self):
		test_queries = []
		for session_query in self.df['cities']:
			test_queries




