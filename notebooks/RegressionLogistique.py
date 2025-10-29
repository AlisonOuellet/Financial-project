# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %%
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.datasets import make_classification

class LogisticRegressionGD:
	def __init__(self, lr=0.1, epochs=1000, fit_intercept=True, verbose=False):
		self.lr = lr # learning rate
		self.epochs = epochs # number of iterations
		self.fit_intercept = fit_intercept # whether to add intercept (bias) (w0)
		self.verbose = verbose #if true, print loss during training
		self.coef_ = None # model parameters (weigths) (w) 

	def _add_intercept(self, X):
		if not self.fit_intercept:
			return X
		intercept = np.ones((X.shape[0], 1))
		return np.hstack((intercept, X))

	def _sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def fit(self, X, y):
		X = np.asarray(X, dtype=float)
		y = np.asarray(y, dtype=float).reshape(-1, 1)
		X = self._add_intercept(X)
		m, n = X.shape
		# weights init
		self.coef_ = np.random.uniform(-0.01, 0.01, size=(n, 1))
		for epoch in range(self.epochs):
			z = X.dot(self.coef_)
			h = self._sigmoid(z)
			# gradient
			grad = (X.T.dot(h - y))
			self.coef_ -= self.lr * grad

		return self

	def predict_proba(self, X):
		X = np.asarray(X, dtype=float)
		X = self._add_intercept(X)
		proba = self._sigmoid(X.dot(self.coef_))
		return proba.ravel()

	def predict(self, X, threshold=0.5):
		return (self.predict_proba(X) >= threshold).astype(int)

# Fonctions utilitaires de prétraitement
def load_and_preprocess(path=None, label_col='default', test_size=0.2, random_state=42):
	"""Charge un CSV si disponible sinon génère un jeu synthétique.
	   Retourne X_train, X_test, y_train, y_test et le scaler utilisé."""
	if path and os.path.exists(path):
		df = pd.read_csv(path)
		if label_col not in df.columns:
			raise ValueError(f"Label column '{label_col}' not found in CSV.")
		# Séparer features / label
		y = df[label_col].astype(int)
		X = df.drop(columns=[label_col])
		# Remplir NA et encoder categoricals
		X = X.fillna(X.median(numeric_only=True))
		X = pd.get_dummies(X, drop_first=True)
	else:
		# Générer un jeu synthétique si aucun fichier fourni
		X, y = make_classification(n_samples=2000, n_features=10, n_informative=6,
								   n_redundant=2, n_clusters_per_class=2, random_state=random_state)
		X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
		y = pd.Series(y)

	# Standardisation
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X.values)

	X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=test_size,
														random_state=random_state, stratify=y.values)
	return X_train, X_test, y_train, y_test, scaler

def evaluate_model(model, X_test, y_test):
	y_pred = model.predict(X_test)
	y_prob = model.predict_proba(X_test)
	metrics = {
		'accuracy': float(accuracy_score(y_test, y_pred)),
		'precision': float(precision_score(y_test, y_pred, zero_division=0)),
		'recall': float(recall_score(y_test, y_pred, zero_division=0)),
		'roc_auc': float(roc_auc_score(y_test, y_prob))
	}
	return metrics

def main():
	parser = argparse.ArgumentParser(description="Régression logistique simple pour défaut de crédit")
	parser.add_argument('--data', type=str, default=None, help='Chemin vers le CSV (doit contenir la colonne "default")')
	parser.add_argument('--label', type=str, default='default', help='Nom de la colonne label dans le CSV')
	parser.add_argument('--lr', type=float, default=0.1)
	parser.add_argument('--epochs', type=int, default=1000)
	parser.add_argument('--verbose', action='store_true')
	args = parser.parse_args()

	X_train, X_test, y_train, y_test, scaler = load_and_preprocess(args.data, label_col=args.label)
	model = LogisticRegressionGD(lr=args.lr, epochs=args.epochs, verbose=args.verbose)
	model.fit(X_train, y_train)
	metrics = evaluate_model(model, X_test, y_test)
	print("Évaluation du modèle:")
	for k, v in metrics.items():
		print(f"  {k}: {v:.4f}")

	# Exemple de prédiction sur quelques instances
	sample_proba = model.predict_proba(X_test[:5])
	sample_pred = model.predict(X_test[:5])
	print("Exemples (proba, préd):")
	for p, pr in zip(sample_proba, sample_pred):
		print(f"  {p:.3f}, {pr}")

if __name__ == "__main__":
	main()
