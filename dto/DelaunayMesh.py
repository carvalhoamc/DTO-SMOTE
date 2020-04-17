from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling.base import BaseOverSampler
from scipy.spatial import Delaunay
from sklearn.utils import check_random_state

from TetrahedronStatistics import tetrahedron_stats
from folders import mesh_folder


class DTO(BaseOverSampler):
	
	def __init__(self,
				 dim_redutor,
				 dataset_name,
				 sampling_strategy='auto',
				 alpha=1,
				 n_jobs=6,
				 order='random',
				 diff_alpha=0.5,
				 equal_alpha=1,
				 random_state=None,
	             pca_s=3):
		
		super(DTO, self).__init__(sampling_strategy=sampling_strategy)
		
		self.dim_redutor = dim_redutor
		self.alpha = alpha
		self.n_jobs = n_jobs
		self.mesh = None
		self.order = order
		self.diff_alpha = diff_alpha
		self.equal_alpha = equal_alpha
		self.dataset_name = dataset_name
		self.random_state = random_state
		self.pca_s = pca_s
	
	def _fit_resample(self, X, y):
		return self._sample(X, y)
	
	def set_order(self, order):
		self.order = order
	
	def set_equal_alpha(self, alpha):
		self.equal_alpha = alpha
	
	def set_diff_alpha(self, alpha):
		self.diff_alpha = alpha
	
	def build_mesh_3D(self, X, y):
		self.dim_redutor.set_params(n_components=self.pca_s)
		X_reduced = self.dim_redutor.fit_transform(X)
		self.mesh = Delaunay(X_reduced)
		np.savetxt(mesh_folder + self.dataset_name + '_simplices.csv', self.mesh.simplices)
		np.savetxt(mesh_folder + self.dataset_name + '_points.csv', self.mesh.points)
		self.tetra_props = pd.DataFrame([Counter(y[simplex])
										 for simplex in self.mesh.simplices]).fillna(0) / 4
		self.tetra_props.to_csv(mesh_folder + self.dataset_name + '_tetra_props.csv', index=False)
		self.tetra_stats = pd.DataFrame([tetrahedron_stats(X_reduced[simplex]).values()
										 for simplex in self.mesh.simplices]).fillna(0)
		self.tetra_stats.to_csv(mesh_folder + self.dataset_name + '_tetra_stats.csv', index=False)
		
		
	def weight_tetrahedrons(self, class_sample, order):
		if order == "random":
			w = self.tetra_props[class_sample]
		else:
			w = self.tetra_props[class_sample] * self.tetra_stats[order]#class_sample==1 (minority)
		
		return (w / sum(w))
	
	def beta(self, y_values, class_sample):
		beta = np.ones(len(y_values)) * self.diff_alpha;
		beta[y_values == class_sample] = self.equal_alpha;
		return self.random_state.dirichlet(beta, 1)[0]
	
	def genPoint(self, x_frame, beta):
		'''
		Feature generation oversampling
		:param x_frame: matrix with original X data vector
		:param beta: for numbers in a vector with total sum = 1 for example beta = [0.25, 0.25, 0.25, 0.25]
		:return: new feature vector
		'''
		return np.dot(beta.T, x_frame)
	
	def _sample(self, X, y):
		
		if self.mesh == None:
			self.build_mesh_3D(X, y)  # mesh and properties
		X_resampled = X.copy()
		y_resampled = y.copy()
		self.random_state = check_random_state(self.random_state)
		#class_sample is the label of each class
		#n_samples is the necessary samples to balanced this class
		for class_sample, n_samples in self.sampling_strategy_.items():
			if n_samples == 0:
				continue
			
			probs = self.weight_tetrahedrons(class_sample, self.order)
			indexnotz = np.nonzero(probs.values)
			simplices_to_interpolate = self.random_state.choice(indexnotz[0], n_samples,p=np.take(probs,indexnotz[0],axis=0))
			X_new = np.array(
					[self.genPoint(X[self.mesh.simplices[i]], self.beta(y[self.mesh.simplices[i]], class_sample))
					 for i in simplices_to_interpolate])
			y_new = np.array([class_sample] * n_samples)
			X_resampled = np.vstack((X_resampled, X_new))
			y_resampled = np.hstack((y_resampled, y_new))
		
		return (X_resampled, y_resampled)
