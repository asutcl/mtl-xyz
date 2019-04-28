import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import helpers

class SessionSet:
	def __init__(self, dataframe, country):
		self.df = dataframe
		self.country = country
		self.index = helpers.get_index_for_country(country)
		self.color = helpers.get_color_for_country(country)
		self.session_count = self.df.shape[0]

	def get_time_metrics(self):
		time_by_hours = self.df.loc[:,'time_of_day']//helpers.SECONDS_PER_HOUR
		return pd.Series(time_by_hours.describe())

	def add_time_scatter_to_plot(self, ax):
		scatter_visualisation_y_offset = 0.5
		ys = np.random.random_sample(((self.session_count,),)) - scatter_visualisation_y_offset + self.index
		ax.scatter(ys, self.df.loc[:,'time_of_day']/helpers.SECONDS_PER_HOUR_F, c=self.color)

