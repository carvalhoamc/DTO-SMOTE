import os
from collections import Counter

import numpy as np
import pandas as pd
from imblearn.base import BaseSampler
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize

from DelaunayMesh import DTO
from classifiers import classifiers
from datasetsDelaunay import datasets
from folders import output_dir, graph_folder, work_dir
from parameters import projectors, order, alphas, train_smote_ext
from gsmote import GeometricSMOTE
import warnings

warnings.filterwarnings('ignore')


class FakeSampler(BaseSampler):
	_sampling_type = 'bypass'
	
	def _fit_resample(self, X, y):
		return X, y


class Oversampling:
	
	def __init__(self):
		pass
	
	def converteY(self, Y):
		
		c = np.unique(Y)
		if not np.all(c == [-1, 1]):
			Y[Y == 1] = -1
			Y[Y == 2] = 1
		return Y
	
	def runSMOTEvariationsGen(self, folder):
		"""
		Create files with SMOTE preprocessing and without preprocessing.
		:param datasets: datasets.
		:param folder:   cross-validation folders.
		:return:
		"""
		smote = SMOTE()
		borderline1 = SMOTE(kind='borderline1')
		borderline2 = SMOTE(kind='borderline2')
		smoteSVM = SMOTE(kind='svm')
		geometric_smote = GeometricSMOTE(n_jobs=8)
		
		for dataset in datasets:
			for fold in range(5):
				path = os.path.join(folder, dataset, str(fold), ''.join([dataset, "_train.csv"]))
				train = np.genfromtxt(path, delimiter=',')
				X = train[:, 0:train.shape[1] - 1]
				Y = train[:, train.shape[1] - 1]
				
				# SMOTE
				print("SMOTE..." + dataset)
				X_res, y_res = smote.fit_sample(X, Y)
				y_res = y_res.reshape(len(y_res), 1)
				newdata = np.hstack([X_res, y_res])
				newtrain = pd.DataFrame(np.vstack([train, newdata]))
				newtrain.to_csv(os.path.join(folder, dataset, str(fold), ''.join([dataset, "_SMOTE.csv"])),
				                header=False, index=False)
				# SMOTE BORDERLINE1
				print("Borderline1..." + dataset)
				X_res, y_res = borderline1.fit_sample(X, Y)
				y_res = y_res.reshape(len(y_res), 1)
				newdata = np.hstack([X_res, y_res])
				newtrain = pd.DataFrame(np.vstack([train, newdata]))
				newtrain.to_csv(os.path.join(folder, dataset, str(fold), ''.join([dataset, "_Borderline1.csv"])),
				                header=False, index=False)
				# SMOTE BORDERLINE2
				print("Borderline2..." + dataset)
				X_res, y_res = borderline2.fit_sample(X, Y)
				y_res = y_res.reshape(len(y_res), 1)
				newdata = np.hstack([X_res, y_res])
				newtrain = pd.DataFrame(np.vstack([train, newdata]))
				newtrain.to_csv(os.path.join(folder, dataset, str(fold), ''.join([dataset, "_Borderline2.csv"])),
				                header=False, index=False)
				# SMOTE SVM
				print("SMOTE SVM..." + dataset)
				X_res, y_res = smoteSVM.fit_sample(X, Y)
				y_res = y_res.reshape(len(y_res), 1)
				newdata = np.hstack([X_res, y_res])
				newtrain = pd.DataFrame(np.vstack([train, newdata]))
				newtrain.to_csv(os.path.join(folder, dataset, str(fold), ''.join([dataset, "_smoteSVM.csv"])),
				                header=False, index=False)
				
				# GEOMETRIC SMOTE
				print("GEOMETRIC SMOTE..." + dataset)
				X_res, y_res = geometric_smote.fit_resample(X, Y)
				y_res = y_res.reshape(len(y_res), 1)
				newdata = np.hstack([X_res, y_res])
				newtrain = pd.DataFrame(np.vstack([train, newdata]))
				newtrain.to_csv(os.path.join(folder, dataset, str(fold), ''.join([dataset, "_Geometric_SMOTE.csv"])),
				                header=False, index=False)
	
	def runDelaunayVariationsGen(self, folder):
		
		for dataset in datasets:
			for fold in range(5):
				path = os.path.join(folder, dataset, str(fold), ''.join([dataset, "_train.csv"]))
				train = np.genfromtxt(path, delimiter=',')
				X = train[:, 0:train.shape[1] - 1]
				Y = train[:, train.shape[1] - 1]
				print("DELAUNAY..." + dataset)
				for p in projectors:
					delaunay = DTO(p, dataset, pca_s=3)
					for o in order:
						for a in alphas:
							name = "delaunay_" + p.__class__.__name__ + "_" + o + "_" + str(a)
							delaunay.set_order(o)
							delaunay.set_equal_alpha(a)
							X_res, y_res = delaunay.fit_sample(X, Y)
							y_res = y_res.reshape(len(y_res), 1)
							newdata = np.hstack([X_res, y_res])
							newtrain = pd.DataFrame(np.vstack([train, newdata]))
							newtrain.to_csv(
									os.path.join(folder, dataset, str(fold), ''.join([dataset, "_" + name + ".csv"])),
									header=False, index=False)
	
	def runClassification(self, folder, SMOTE=False):
		print("INIT CLASSIFICATION IMBALANCED DATASETS")
		dfcol = ['ID', 'DATASET', 'FOLD', 'PREPROC', 'ALGORITHM', 'MODE', 'ORDER', 'ALPHA', 'PRE', 'REC', 'SPE', 'F1',
		         'GEO', 'IBA',
		         'AUC']
		df = pd.DataFrame(columns=dfcol)
		i = 0
		
		for dataset in datasets:
			print(dataset)
			for fold in range(5):
				print(fold)
				test_path = os.path.join(folder, dataset, str(fold), ''.join([dataset, "_test.csv"]))
				test = np.genfromtxt(test_path, delimiter=',')
				X_test = test[:, 0:test.shape[1] - 1]
				Y_test = test[:, test.shape[1] - 1]
				Y_test = self.converteY(Y_test)
				
				# SMOTE LIKE CLASSIFICATION
				if SMOTE == True:
					print("RUN SMOTE LIKE")
					for ext in train_smote_ext:
						train_path = os.path.join(folder, dataset, str(fold), ''.join([dataset, ext, ".csv"]))
						train = np.genfromtxt(train_path, delimiter=',')
						X_train = train[:, 0:train.shape[1] - 1]
						Y_train = train[:, train.shape[1] - 1]
						Y_train = self.converteY(Y_train)  # Biclass only
						
						if ext == "_train":
							X, Y = X_train, Y_train  # original dataset for plotting
						for name, clf in classifiers.items():
							clf.fit(X_train, Y_train)
							Y_pred = clf.predict(X_test)
							res = classification_report_imbalanced(Y_test, Y_pred)
							identificador = dataset + '_' + ext + '_' + name
							aux = res.split()
							score = aux[-7:-1]
							df.at[i, 'ID'] = identificador
							df.at[i, 'DATASET'] = dataset
							df.at[i, 'FOLD'] = fold
							df.at[i, 'PREPROC'] = ext
							df.at[i, 'ALGORITHM'] = name
							df.at[i, 'MODE'] = 'PCA'
							df.at[i, 'ORDER'] = 'NONE'
							df.at[i, 'ALPHA'] = 'NONE'
							df.at[i, 'PRE'] = score[0]
							df.at[i, 'REC'] = score[1]
							df.at[i, 'SPE'] = score[2]
							df.at[i, 'F1'] = score[3]
							df.at[i, 'GEO'] = score[4]
							df.at[i, 'IBA'] = score[5]
							df.at[i, 'AUC'] = roc_auc_score(Y_test, Y_pred)  # biclass
							# df.at[i, 'AUC'] = -1  # multiclass
							
							i = i + 1
				
				# DELAUNAY LIKE CLASSIFICATION
				print("Run DTO")
				for p in projectors:
					for o in order:
						for a in alphas:
							id = "_delaunay_" + p.__class__.__name__ + "_" + o + "_" + str(a)
							train_path = os.path.join(folder, dataset, str(fold), ''.join([dataset, id, ".csv"]))
							train = np.genfromtxt(train_path, delimiter=',')
							X_train = train[:, 0:train.shape[1] - 1]
							Y_train = train[:, train.shape[1] - 1]
							Y_train = self.converteY(Y_train)  # multiclass
							for alg, clf in classifiers.items():
								clf.fit(X_train, Y_train)
								Y_pred = clf.predict(X_test)
								res = classification_report_imbalanced(Y_test, Y_pred)
								identificador = dataset + '_' + id + '_' + alg
								aux = res.split()
								score = aux[-7:-1]
								df.at[i, 'ID'] = identificador
								df.at[i, 'DATASET'] = dataset
								df.at[i, 'FOLD'] = fold
								df.at[i, 'PREPROC'] = '_delaunay' + "_" + o + "_" + str(a)
								df.at[i, 'ALGORITHM'] = alg
								df.at[i, 'MODE'] = p.__class__.__name__
								df.at[i, 'ORDER'] = o
								df.at[i, 'ALPHA'] = a
								df.at[i, 'PRE'] = score[0]
								df.at[i, 'REC'] = score[1]
								df.at[i, 'SPE'] = score[2]
								df.at[i, 'F1'] = score[3]
								df.at[i, 'GEO'] = score[4]
								df.at[i, 'IBA'] = score[5]
								df.at[i, 'AUC'] = roc_auc_score(Y_test, Y_pred)  # biclass
								# df.at[i, 'AUC'] = -1  # multiclass
								i = i + 1
			
			df.to_csv(output_dir + 'resultado_biclasse_' + p.__class__.__name__ + '.csv', index=False)
			print('DTO file on SSD')
	
	def createValidationData(self, folder):
		"""
		Create sub datasets for cross validation purpose
		:param datasets: List of datasets
		:param folder: Where datasets was stored
		:return:
		"""
		
		for dataset in datasets:
			print(dataset)
			fname = os.path.join(folder, ''.join([dataset, ".npz"]))
			data = np.load(fname)
			X = normalize(data['arr_0'])
			Y = np.array(data['arr_1'])
			skf = StratifiedKFold(n_splits=5, shuffle=True)
			
			for fold, (train_index, test_index) in enumerate(skf.split(X, Y)):
				X_train, X_test = X[train_index], X[test_index]
				y_train, y_test = data['arr_1'][train_index], data['arr_1'][test_index]
				y_train = y_train.reshape(len(y_train), 1)
				y_test = y_test.reshape(len(y_test), 1)
				train = pd.DataFrame(np.hstack((X_train, y_train)))
				test = pd.DataFrame(np.hstack((X_test, y_test)))
				os.makedirs(os.path.join(folder, dataset, str(fold)))
				train.to_csv(os.path.join(folder, dataset, str(fold), ''.join([dataset, "_train.csv"])), header=False,
				             index=False)
				test.to_csv(os.path.join(folder, dataset, str(fold), ''.join([dataset, "_test.csv"])), header=False,
				            index=False)
