import pandas as pd
import numpy as np
from SessionSet import SessionSet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer

class DummyModel:

	def predict_proba(self, *args):
		return np.array([[0.5, 0.5]])

class LogisticPrior:

	def __init__(self, df, columns, targets,epsilon=1e-5):
		self.epsilon = 1e-5
		self.columns = [int(df.columns.get_loc(x)) for x in columns]
		self.target_list = targets
		self.targets = {}
		for target in targets:
			self.targets[target] = DummyModel()

	def fit(self, df):
		X = df.iloc[:,self.columns].values.astype(np.float)
		self.scaler = MinMaxScaler()
		self.scaler.fit(X)
		X = self.scaler.transform(X)
		for target in self.target_list:
			y = df.apply(lambda row: target in row['cities'], axis=1).values.astype(np.float)
			if np.sum(y) <= 3:
				self.targets[target] = DummyModel()
			else:
				ftwo_scorer = make_scorer(fbeta_score, beta=2)
				param_grid = [
					{
						'penalty' : ['l2'],
						'C' : np.logspace(-8, 4, 40),
						'solver' : ['liblinear'],
						 'class_weight': ['balanced']
					}
				]
				clf = GridSearchCV(LogisticRegression(), param_grid = param_grid, cv = 3, verbose=False, n_jobs=-1, scoring=ftwo_scorer)
				self.targets[target] = clf.fit(X, y)

	def predict(self, session):
		X = np.array([session[self.columns]], dtype=np.float)
		X = self.scaler.transform(X)
		prior_lr = np.zeros((1,len(self.target_list)))
		for i, target in enumerate(self.target_list):
			prior_lr[0,i] = self.targets[target].predict_proba(X)[0,1]
		prior_lr = np.log(prior_lr)
		prior_lr[np.isnan(prior_lr)] = self.epsilon
		prior_lr[prior_lr == -np.inf] = self.epsilon
		return prior_lr

