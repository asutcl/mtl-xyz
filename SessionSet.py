import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import helpers

class SessionSet:
	"""
	Class that stores the sessions dataframe and adds useful functions
	to interprete the data.
	"""
	def __init__(self, dataframe, country, epsilon=1e-5):
		self.df = dataframe
		self.country = country
		self.index = helpers.get_index_for_country(country)
		self.color = helpers.get_color_for_country(country)
		self.session_count = self.df.shape[0]
		self.unique_users = set(self.df['user.user_id'].unique())
		self.epsilon = epsilon

	def get_time_metrics(self):
		time_by_hours = self.df.loc[:,'hour_of_day']
		return pd.Series(time_by_hours.describe())

	def add_time_scatter_to_plot(self, ax):
		scatter_visualisation_y_offset = 0.5
		ys = np.random.random_sample(((self.session_count,),)) - scatter_visualisation_y_offset + self.index
		ax.scatter(ys, self.df.loc[:,'time_of_day']/helpers.SECONDS_PER_HOUR_F, c=self.color)

	def get_requests_stats_per_user(self):
		sdf = self.df[['cities','user.user_id']]
		users_grouped = sdf.groupby(['user.user_id']).agg(lambda requests: requests.tolist())
		users_grouped['cities_flat'] = users_grouped.apply(lambda row: helpers.flatten_list_of_lists(row['cities']), axis=1)
		users_grouped['cities_per_user'] = users_grouped.apply(lambda row: len(row['cities_flat']), axis=1)
		return users_grouped

	def get_requests_stats(self):
		sdf = self.df[['cities','user.user_id']]
		return sdf.apply(lambda row: len(row['cities']), axis=1)

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

	def get_top_5_cities_with_precentage(self):
		return self.get_top_cities_with_precentage(5)

	def get_top_cities_with_precentage(self, top):
		occurences = self.get_cities_occurence_count();
		for city in occurences.keys():
			occurences[city] /= self.session_count
		sorted_occurences = sorted(occurences.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)
		return sorted_occurences[0:top]








