import os
from collections import Counter
from os import listdir
from os.path import isfile, join
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

style.use('ggplot')
import scipy
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
import numpy as np
from sys import argv
import Orange
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors, cm
from matplotlib.collections import PolyCollection
from classifiers import classifiers_list
from datasetsDelaunay import dataset_list_bi, dataset_list_mult
from folders import output_dir, dir_pca_biclasse, metricas_biclasse, dir_pca_multiclasse, metricas_multiclasse
from parameters import order, alphas

order_dict = {'area': 1,
              'volume': 2,
              'area_volume_ratio': 3,
              'edge_ratio': 4,
              'radius_ratio': 5,
              'aspect_ratio': 6,
              'max_solid_angle': 7,
              'min_solid_angle': 8,
              'solid_angle': 9}


class Statistics:
	
	def __init__(self):
		pass
	
	def compute_CD_customizado(self, avranks, n, alpha="0.05", test="nemenyi"):
		"""
		Returns critical difference for Nemenyi or Bonferroni-Dunn test
		according to given alpha (either alpha="0.05" or alpha="0.1") for average
		ranks and number of tested datasets N. Test can be either "nemenyi" for
		for Nemenyi two tailed test or "bonferroni-dunn" for Bonferroni-Dunn test.
		"""
		k = len(avranks)
		d = {("nemenyi", "0.05"): [1.960, 2.344, 2.569, 2.728, 2.850, 2.948, 3.031, 3.102, 3.164, 3.219, 3.268, 3.313,
		                           3.354, 3.391, 3.426,
		                           3.458, 3.489, 3.517, 3.544, 3.569, 3.593, 3.616, 3.637, 3.658, 3.678, 3.696, 3.714,
		                           3.732, 3.749, 3.765,
		                           3.780, 3.795, 3.810, 3.824, 3.837, 3.850, 3.863, 3.876, 3.888, 3.899, 3.911, 3.922,
		                           3.933, 3.943, 3.954,
		                           3.964, 3.973, 3.983, 3.992],
		     ("nemenyi", "0.1"): [0, 0, 1.644854, 2.052293, 2.291341, 2.459516,
		                          2.588521, 2.692732, 2.779884, 2.854606, 2.919889,
		                          2.977768, 3.029694, 3.076733, 3.119693, 3.159199,
		                          3.195743, 3.229723, 3.261461, 3.291224, 3.319233],
		     ("bonferroni-dunn", "0.05"): [0, 0, 1.960, 2.241, 2.394, 2.498, 2.576,
		                                   2.638, 2.690, 2.724, 2.773],
		     ("bonferroni-dunn", "0.1"): [0, 0, 1.645, 1.960, 2.128, 2.241, 2.326,
		                                  2.394, 2.450, 2.498, 2.539]}
		q = d[(test, alpha)]
		cd = q[k] * (k * (k + 1) / (6.0 * n)) ** 0.5
		return cd
	
	def calcula_media_folds_biclasse(self, df):
		t = pd.Series(data=np.arange(0, df.shape[0], 1))
		dfr = pd.DataFrame(
				columns=['MODE', 'DATASET', 'PREPROC', 'ALGORITHM', 'ORDER', 'ALPHA', 'PRE', 'REC', 'SPE', 'F1', 'GEO',
				         'IBA', 'AUC'],
				index=np.arange(0, int(t.shape[0] / 5)))
		
		df_temp = df.groupby(by=['MODE', 'DATASET', 'PREPROC', 'ALGORITHM'])
		idx = dfr.index.values
		i = idx[0]
		for name, group in df_temp:
			group = group.reset_index()
			dfr.at[i, 'MODE'] = group.loc[0, 'MODE']
			mode = group.loc[0, 'MODE']
			dfr.at[i, 'DATASET'] = group.loc[0, 'DATASET']
			dfr.at[i, 'PREPROC'] = group.loc[0, 'PREPROC']
			dfr.at[i, 'ALGORITHM'] = group.loc[0, 'ALGORITHM']
			dfr.at[i, 'ORDER'] = group.loc[0, 'ORDER']
			dfr.at[i, 'ALPHA'] = group.loc[0, 'ALPHA']
			dfr.at[i, 'PRE'] = group['PRE'].mean()
			dfr.at[i, 'REC'] = group['REC'].mean()
			dfr.at[i, 'SPE'] = group['SPE'].mean()
			dfr.at[i, 'F1'] = group['F1'].mean()
			dfr.at[i, 'GEO'] = group['GEO'].mean()
			dfr.at[i, 'IBA'] = group['IBA'].mean()
			dfr.at[i, 'AUC'] = group['AUC'].mean()
			i = i + 1
			print(i)
		
		dfr.to_csv(output_dir + 'resultado_media_biclasse_' + mode + '.csv', index=False)
	
	def calcula_media_folds_multiclass(self, df):
		t = pd.Series(data=np.arange(0, df.shape[0], 1))
		dfr = pd.DataFrame(
				columns=['MODE', 'DATASET', 'PREPROC', 'ALGORITHM', 'ORDER', 'ALPHA', 'PRE', 'REC', 'SPE', 'F1', 'GEO',
				         'IBA'],
				index=np.arange(0, int(t.shape[0] / 5)))
		
		df_temp = df.groupby(by=['MODE', 'DATASET', 'PREPROC', 'ALGORITHM'])
		idx = dfr.index.values
		i = idx[0]
		for name, group in df_temp:
			group = group.reset_index()
			dfr.at[i, 'MODE'] = group.loc[0, 'MODE']
			mode = group.loc[0, 'MODE']
			dfr.at[i, 'DATASET'] = group.loc[0, 'DATASET']
			dfr.at[i, 'PREPROC'] = group.loc[0, 'PREPROC']
			dfr.at[i, 'ALGORITHM'] = group.loc[0, 'ALGORITHM']
			dfr.at[i, 'ORDER'] = group.loc[0, 'ORDER']
			dfr.at[i, 'ALPHA'] = group.loc[0, 'ALPHA']
			dfr.at[i, 'PRE'] = group['PRE'].mean()
			dfr.at[i, 'REC'] = group['REC'].mean()
			dfr.at[i, 'SPE'] = group['SPE'].mean()
			dfr.at[i, 'F1'] = group['F1'].mean()
			dfr.at[i, 'GEO'] = group['GEO'].mean()
			dfr.at[i, 'IBA'] = group['IBA'].mean()
			i = i + 1
			print(i)
		
		dfr.to_csv(output_dir + 'resultado_media_multiclass_' + mode + '.csv', index=False)
	
	def separa_delaunay_biclass(self, filename):
		df = pd.read_csv(filename)
		list_base = []
		for p in np.arange(0, len(preproc_type)):
			list_base.append(df[(df['PREPROC'] == preproc_type[p])])
		df_base = list_base.pop(0)
		for i in np.arange(0, len(list_base)):
			df_base = pd.concat([df_base, list_base[i]], ignore_index=True)
		
		for o in order:
			for a in alphas:
				dfr = df[(df['ORDER'] == o)]
				dfr1 = dfr[(dfr['ALPHA'] == str(a))]
				df_file = pd.concat([df_base, dfr1], ignore_index=True)
				df_file.to_csv('./../output_dir/result_biclass' + '_' + o + '_' + str(a) + '.csv', index=False)
	
	def read_dir_files(self, dir_name):
		f = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
		return f
	
	def find_best_rank(self, results_dir, tipo):
		results = self.read_dir_files(results_dir)
		df = pd.DataFrame(columns=[['ARQUIVO', 'WINER']])
		i = 0
		for f in results:
			df_temp = pd.read_csv(results_dir + f)
			df.at[i, 'ARQUIVO'] = f
			df.at[i, 'WINER'] = df_temp.iloc[0, 0]
			i += 1
		
		df.to_csv(output_dir + tipo)
	
	def find_best_delaunay(self, results_dir, tipo):
		df = pd.read_csv(results_dir + tipo)
		i = 0
		j = 0
		df_best = pd.DataFrame(columns=['ARQUIVO', 'WINER'])
		win = list(df['WINER'])
		for w in win:
			if w == 'DELAUNAY':
				df_best.at[i, 'ARQUIVO'] = df.iloc[j, 1]
				df_best.at[i, 'WINER'] = df.iloc[j, 2]
				i += 1
			j += 1
		
		df_best.to_csv(output_dir + 'only_best_delaunay_pca_biclass_media_rank.csv')
	
	def rank_by_algorithm(self, df, tipo, wd, reducao, order, alpha):
		'''
		Calcula rank
		:param df:
		:param tipo:
		:param wd:
		:param delaunay_type:
		:return:
		'''
		
		df_tabela = pd.DataFrame(
				columns=['DATASET', 'ALGORITHM', 'ORIGINAL', 'RANK_ORIGINAL', 'SMOTE', 'RANK_SMOTE', 'SMOTE_SVM',
				         'RANK_SMOTE_SVM', 'BORDERLINE1', 'RANK_BORDERLINE1', 'BORDERLINE2', 'RANK_BORDERLINE2',
				         'GEOMETRIC_SMOTE', 'RANK_GEOMETRIC_SMOTE',
				         'DELAUNAY', 'RANK_DELAUNAY', 'DELAUNAY_TYPE', 'ALPHA', 'unit'])
		
		df_temp = df.groupby(by=['ALGORITHM'])
		for name, group in df_temp:
			group = group.reset_index()
			group.drop('index', axis=1, inplace=True)
			#df.to_csv(dir_pca_biclasse + reducao + '_' + tipo + '_' + order + '_' + str(alpha) + '.csv')
			df.to_csv(dir_pca_multiclasse + reducao + '_' + tipo + '_' + order + '_' + str(alpha) + '.csv')
			
			j = 0
			#for d in dataset_list_mult:
			for d in dataset_list_bi:
				#for m in metricas_multiclasse:
				for m in metricas_biclasse:
					aux = group[group['DATASET'] == d]
					aux = aux.reset_index()
					df_tabela.at[j, 'DATASET'] = d
					df_tabela.at[j, 'ALGORITHM'] = name
					indice = aux.PREPROC[aux.PREPROC == '_train'].index.tolist()[0]
					df_tabela.at[j, 'ORIGINAL'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.PREPROC == '_SMOTE'].index.tolist()[0]
					df_tabela.at[j, 'SMOTE'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.PREPROC == '_smoteSVM'].index.tolist()[0]
					df_tabela.at[j, 'SMOTE_SVM'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.PREPROC == '_Borderline1'].index.tolist()[0]
					df_tabela.at[j, 'BORDERLINE1'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.PREPROC == '_Borderline2'].index.tolist()[0]
					df_tabela.at[j, 'BORDERLINE2'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.PREPROC == '_Geometric_SMOTE'].index.tolist()[0]
					df_tabela.at[j, 'GEOMETRIC_SMOTE'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.ORDER == order].index.tolist()[0]
					df_tabela.at[j, 'DELAUNAY'] = aux.at[indice, m]
					df_tabela.at[j, 'DELAUNAY_TYPE'] = order
					df_tabela.at[j, 'ALPHA'] = alpha
					df_tabela.at[j, 'unit'] = m
					j += 1
			
			df_pre = df_tabela[df_tabela['unit'] == 'PRE']
			df_rec = df_tabela[df_tabela['unit'] == 'REC']
			df_spe = df_tabela[df_tabela['unit'] == 'SPE']
			df_f1 = df_tabela[df_tabela['unit'] == 'F1']
			df_geo = df_tabela[df_tabela['unit'] == 'GEO']
			df_iba = df_tabela[df_tabela['unit'] == 'IBA']
			df_auc = df_tabela[df_tabela['unit'] == 'AUC']
			
			pre = df_pre[
				['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE', 'DELAUNAY']]
			rec = df_rec[
				['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE', 'DELAUNAY']]
			spe = df_spe[
				['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE', 'DELAUNAY']]
			f1 = df_f1[['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE', 'DELAUNAY']]
			geo = df_geo[
				['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE', 'DELAUNAY']]
			iba = df_iba[
				['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE', 'DELAUNAY']]
			auc = df_auc[
				['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE', 'DELAUNAY']]
			
			pre = pre.reset_index()
			pre.drop('index', axis=1, inplace=True)
			rec = rec.reset_index()
			rec.drop('index', axis=1, inplace=True)
			spe = spe.reset_index()
			spe.drop('index', axis=1, inplace=True)
			f1 = f1.reset_index()
			f1.drop('index', axis=1, inplace=True)
			geo = geo.reset_index()
			geo.drop('index', axis=1, inplace=True)
			iba = iba.reset_index()
			iba.drop('index', axis=1, inplace=True)
			auc = auc.reset_index()
			auc.drop('index', axis=1, inplace=True)
			
			# calcula rank linha a linha
			pre_rank = pre.rank(axis=1, ascending=False)
			rec_rank = rec.rank(axis=1, ascending=False)
			spe_rank = spe.rank(axis=1, ascending=False)
			f1_rank = f1.rank(axis=1, ascending=False)
			geo_rank = geo.rank(axis=1, ascending=False)
			iba_rank = iba.rank(axis=1, ascending=False)
			auc_rank = auc.rank(axis=1, ascending=False)
			
			df_pre = df_pre.reset_index()
			df_pre.drop('index', axis=1, inplace=True)
			df_pre['RANK_ORIGINAL'] = pre_rank['ORIGINAL']
			df_pre['RANK_SMOTE'] = pre_rank['SMOTE']
			df_pre['RANK_SMOTE_SVM'] = pre_rank['SMOTE_SVM']
			df_pre['RANK_BORDERLINE1'] = pre_rank['BORDERLINE1']
			df_pre['RANK_BORDERLINE2'] = pre_rank['BORDERLINE2']
			df_pre['RANK_GEOMETRIC_SMOTE'] = pre_rank['GEOMETRIC_SMOTE']
			df_pre['RANK_DELAUNAY'] = pre_rank['DELAUNAY']
			
			df_rec = df_rec.reset_index()
			df_rec.drop('index', axis=1, inplace=True)
			df_rec['RANK_ORIGINAL'] = rec_rank['ORIGINAL']
			df_rec['RANK_SMOTE'] = rec_rank['SMOTE']
			df_rec['RANK_SMOTE_SVM'] = rec_rank['SMOTE_SVM']
			df_rec['RANK_BORDERLINE1'] = rec_rank['BORDERLINE1']
			df_rec['RANK_BORDERLINE2'] = rec_rank['BORDERLINE2']
			df_rec['RANK_GEOMETRIC_SMOTE'] = rec_rank['GEOMETRIC_SMOTE']
			df_rec['RANK_DELAUNAY'] = rec_rank['DELAUNAY']
			
			df_spe = df_spe.reset_index()
			df_spe.drop('index', axis=1, inplace=True)
			df_spe['RANK_ORIGINAL'] = spe_rank['ORIGINAL']
			df_spe['RANK_SMOTE'] = spe_rank['SMOTE']
			df_spe['RANK_SMOTE_SVM'] = spe_rank['SMOTE_SVM']
			df_spe['RANK_BORDERLINE1'] = spe_rank['BORDERLINE1']
			df_spe['RANK_BORDERLINE2'] = spe_rank['BORDERLINE2']
			df_spe['RANK_GEOMETRIC_SMOTE'] = spe_rank['GEOMETRIC_SMOTE']
			df_spe['RANK_DELAUNAY'] = spe_rank['DELAUNAY']
			
			df_f1 = df_f1.reset_index()
			df_f1.drop('index', axis=1, inplace=True)
			df_f1['RANK_ORIGINAL'] = f1_rank['ORIGINAL']
			df_f1['RANK_SMOTE'] = f1_rank['SMOTE']
			df_f1['RANK_SMOTE_SVM'] = f1_rank['SMOTE_SVM']
			df_f1['RANK_BORDERLINE1'] = f1_rank['BORDERLINE1']
			df_f1['RANK_BORDERLINE2'] = f1_rank['BORDERLINE2']
			df_f1['RANK_GEOMETRIC_SMOTE'] = f1_rank['GEOMETRIC_SMOTE']
			df_f1['RANK_DELAUNAY'] = f1_rank['DELAUNAY']
			
			df_geo = df_geo.reset_index()
			df_geo.drop('index', axis=1, inplace=True)
			df_geo['RANK_ORIGINAL'] = geo_rank['ORIGINAL']
			df_geo['RANK_SMOTE'] = geo_rank['SMOTE']
			df_geo['RANK_SMOTE_SVM'] = geo_rank['SMOTE_SVM']
			df_geo['RANK_BORDERLINE1'] = geo_rank['BORDERLINE1']
			df_geo['RANK_BORDERLINE2'] = geo_rank['BORDERLINE2']
			df_geo['RANK_GEOMETRIC_SMOTE'] = geo_rank['GEOMETRIC_SMOTE']
			df_geo['RANK_DELAUNAY'] = geo_rank['DELAUNAY']
			
			df_iba = df_iba.reset_index()
			df_iba.drop('index', axis=1, inplace=True)
			df_iba['RANK_ORIGINAL'] = iba_rank['ORIGINAL']
			df_iba['RANK_SMOTE'] = iba_rank['SMOTE']
			df_iba['RANK_SMOTE_SVM'] = iba_rank['SMOTE_SVM']
			df_iba['RANK_BORDERLINE1'] = iba_rank['BORDERLINE1']
			df_iba['RANK_BORDERLINE2'] = iba_rank['BORDERLINE2']
			df_iba['RANK_GEOMETRIC_SMOTE'] = iba_rank['GEOMETRIC_SMOTE']
			df_iba['RANK_DELAUNAY'] = iba_rank['DELAUNAY']
			
			df_auc = df_auc.reset_index()
			df_auc.drop('index', axis=1, inplace=True)
			df_auc['RANK_ORIGINAL'] = auc_rank['ORIGINAL']
			df_auc['RANK_SMOTE'] = auc_rank['SMOTE']
			df_auc['RANK_SMOTE_SVM'] = auc_rank['SMOTE_SVM']
			df_auc['RANK_BORDERLINE1'] = auc_rank['BORDERLINE1']
			df_auc['RANK_BORDERLINE2'] = auc_rank['BORDERLINE2']
			df_auc['RANK_GEOMETRIC_SMOTE'] = auc_rank['GEOMETRIC_SMOTE']
			df_auc['RANK_DELAUNAY'] = auc_rank['DELAUNAY']
			
			# avarege rank
			media_pre_rank = pre_rank.mean(axis=0)
			media_rec_rank = rec_rank.mean(axis=0)
			media_spe_rank = spe_rank.mean(axis=0)
			media_f1_rank = f1_rank.mean(axis=0)
			media_geo_rank = geo_rank.mean(axis=0)
			media_iba_rank = iba_rank.mean(axis=0)
			media_auc_rank = auc_rank.mean(axis=0)
			
			media_pre_rank_file = media_pre_rank.reset_index()
			media_pre_rank_file = media_pre_rank_file.sort_values(by=0)
			
			media_rec_rank_file = media_rec_rank.reset_index()
			media_rec_rank_file = media_rec_rank_file.sort_values(by=0)
			
			media_spe_rank_file = media_spe_rank.reset_index()
			media_spe_rank_file = media_spe_rank_file.sort_values(by=0)
			
			media_f1_rank_file = media_f1_rank.reset_index()
			media_f1_rank_file = media_f1_rank_file.sort_values(by=0)
			
			media_geo_rank_file = media_geo_rank.reset_index()
			media_geo_rank_file = media_geo_rank_file.sort_values(by=0)
			
			media_iba_rank_file = media_iba_rank.reset_index()
			media_iba_rank_file = media_iba_rank_file.sort_values(by=0)
			
			media_auc_rank_file = media_auc_rank.reset_index()
			media_auc_rank_file = media_auc_rank_file.sort_values(by=0)
			
			# Grava arquivos importantes
			df_pre.to_csv(
					wd + 'total_rank/' + reducao + '_total_rank_' + tipo + '_' + order + '_' + str(
							alpha) + '_' + name + '_pre.csv',
					index=False)
			df_rec.to_csv(
					wd + 'total_rank/' + reducao + '_total_rank_' + tipo + '_' + order + '_' + str(
							alpha) + '_' + name + '_rec.csv',
					index=False)
			df_spe.to_csv(
					wd + 'total_rank/' + reducao + '_total_rank_' + tipo + '_' + order + '_' + str(
							alpha) + '_' + name + '_spe.csv',
					index=False)
			df_f1.to_csv(wd + 'total_rank/' + reducao + '_total_rank_' + tipo + '_' + order + '_' + str(
					alpha) + '_' + name + '_f1.csv',
			             index=False)
			df_geo.to_csv(
					wd + 'total_rank/' + reducao + '_total_rank_' + tipo + '_' + order + '_' + str(
							alpha) + '_' + name + '_geo.csv',
					index=False)
			df_iba.to_csv(
					wd + 'total_rank/' + reducao + '_total_rank_' + tipo + '_' + order + '_' + str(
							alpha) + '_' + name + '_iba.csv',
					index=False)
			df_auc.to_csv(
					wd + 'total_rank/' + reducao + '_total_rank_' + tipo + '_' + order + '_' + str(
							alpha) + '_' + name + '_auc.csv',
					index=False)
			
			media_pre_rank_file.to_csv(
					wd + 'media_rank/' + reducao + '_media_rank_' + tipo + '_' + order + '_' + str(
							alpha) + '_' + name + '_pre.csv',
					index=False)
			media_rec_rank_file.to_csv(
					wd + 'media_rank/' + reducao + '_media_rank_' + tipo + '_' + order + '_' + str(
							alpha) + '_' + name + '_rec.csv',
					index=False)
			media_spe_rank_file.to_csv(
					wd + 'media_rank/' + reducao + '_media_rank_' + tipo + '_' + order + '_' + str(
							alpha) + '_' + name + '_spe.csv',
					index=False)
			media_f1_rank_file.to_csv(
					wd + 'media_rank/' + reducao + '_media_rank_' + tipo + '_' + order + '_' + str(
							alpha) + '_' + name + '_f1.csv',
					index=False)
			media_geo_rank_file.to_csv(
					wd + 'media_rank/' + reducao + '_media_rank_' + tipo + '_' + order + '_' + str(
							alpha) + '_' + name + '_geo.csv',
					index=False)
			media_iba_rank_file.to_csv(
					wd + 'media_rank/' + reducao + '_media_rank_' + tipo + '_' + order + '_' + str(
							alpha) + '_' + name + '_iba.csv',
					index=False)
			media_auc_rank_file.to_csv(
					wd + 'media_rank/' + reducao + '_media_rank_' + tipo + '_' + order + '_' + str(
							alpha) + '_' + name + '_auc.csv',
					index=False)
			
			delaunay_type = order + '_' + str(alpha)
			
			# grafico CD
			identificadores = ['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE',
			                   delaunay_type]
			avranks = list(media_pre_rank)
			cd = Orange.evaluation.compute_CD(avranks, len(dataset_list_bi))
			Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
			plt.savefig(
					wd + 'figurasCD/' + 'cd_' + reducao + '_' + tipo + '_' + delaunay_type + '_' + name + '_pre.pdf')
			plt.close()
			
			avranks = list(media_rec_rank)
			cd = Orange.evaluation.compute_CD(avranks, len(dataset_list_bi))
			Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
			plt.savefig(
					wd + 'figurasCD/' + 'cd_' + reducao + '_' + tipo + '_' + delaunay_type + '_' + name + '_rec.pdf')
			plt.close()
			
			avranks = list(media_spe_rank)
			cd = Orange.evaluation.compute_CD(avranks, len(dataset_list_bi))
			Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
			plt.savefig(
					wd + 'figurasCD/' + 'cd_' + reducao + '_' + tipo + '_' + delaunay_type + '_' + name + '_spe.pdf')
			plt.close()
			
			avranks = list(media_f1_rank)
			cd = Orange.evaluation.compute_CD(avranks, len(dataset_list_bi))
			Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
			plt.savefig(wd + 'figurasCD/' + 'cd_' + reducao + '_' + tipo + '_' + delaunay_type + '_' + name + '_f1.pdf')
			plt.close()
			
			avranks = list(media_geo_rank)
			cd = Orange.evaluation.compute_CD(avranks, len(dataset_list_bi))
			Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
			plt.savefig(
					wd + 'figurasCD/' + 'cd_' + reducao + '_' + tipo + '_' + delaunay_type + '_' + name + '_geo.pdf')
			plt.close()
			
			avranks = list(media_iba_rank)
			cd = Orange.evaluation.compute_CD(avranks, len(dataset_list_bi))
			Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
			plt.savefig(
					wd + 'figurasCD/' + 'cd_' + reducao + '_' + tipo + '_' + delaunay_type + '_' + name + '_iba.pdf')
			plt.close()
			
			avranks = list(media_auc_rank)
			cd = Orange.evaluation.compute_CD(avranks, len(dataset_list_bi))
			Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
			plt.savefig(
					wd + 'figurasCD/' + 'cd_' + reducao + '_' + tipo + '_' + delaunay_type + '_' + name + '_auc.pdf')
			plt.close()
			
			print('Delaunay Type= ', delaunay_type)
			print('Algorithm= ', name)
	
	def rank_total_by_algorithm(self, tipo, wd, reducao, order, alpha):
		delaunay_name = 'RANK_DTO_' + str(order) + '_' + str(alpha)
		cols = ['ALGORITHM', 'RANK_ORIGINAL', 'RANK_SMOTE', 'RANK_SMOTE_SVM', 'RANK_BORDERLINE1',
		        'RANK_BORDERLINE2', 'RANK_GEOMETRIC_SMOTE', 'RANK_DELAUNAY']
		for name in classifiers_list:
			print(os.path.abspath(os.getcwd()))
			# Grava arquivos importantes
			path_name = wd + reducao + '_total_rank_' + tipo + '_' + order + '_' + str(alpha) + '_' + name + '_pre.csv'
			df_pre = pd.read_csv(path_name)
			path_name = wd + reducao + '_total_rank_' + tipo + '_' + order + '_' + str(alpha) + '_' + name + '_rec.csv'
			df_rec = pd.read_csv(path_name)
			path_name = wd + reducao + '_total_rank_' + tipo + '_' + order + '_' + str(alpha) + '_' + name + '_spe.csv'
			df_spe = pd.read_csv(path_name)
			path_name = wd + reducao + '_total_rank_' + tipo + '_' + order + '_' + str(alpha) + '_' + name + '_f1.csv'
			df_f1 = pd.read_csv(path_name)
			path_name = wd + reducao + '_total_rank_' + tipo + '_' + order + '_' + str(alpha) + '_' + name + '_geo.csv'
			df_geo = pd.read_csv(path_name)
			path_name = wd + reducao + '_total_rank_' + tipo + '_' + order + '_' + str(alpha) + '_' + name + '_iba.csv'
			df_iba = pd.read_csv(path_name)
			path_name = wd + reducao + '_total_rank_' + tipo + '_' + order + '_' + str(alpha) + '_' + name + '_auc.csv'
			df_auc = pd.read_csv(path_name)
			
			# PRE
			df_pre_col = df_pre[cols]
			df_pre_col.loc[:, delaunay_name] = df_pre_col['RANK_DELAUNAY'].values
			df_pre_col.drop(['RANK_DELAUNAY'], axis=1, inplace=True)
			ranking_pre = df_pre_col.groupby(['ALGORITHM']).mean()
			path_name = wd + reducao + '_rank_by_algorithm_' + tipo + '_' + order + '_' + str(
					alpha) + '_' + name + '_pre.csv'
			ranking_pre['ALGORITHM'] = name
			ranking_pre.to_csv(path_name, index=False)
			
			# REC
			df_rec_col = df_rec[cols]
			df_rec_col.loc[:, delaunay_name] = df_rec_col['RANK_DELAUNAY'].values
			df_rec_col.drop(['RANK_DELAUNAY'], axis=1, inplace=True)
			ranking_rec = df_rec_col.groupby(['ALGORITHM']).mean()
			path_name = wd + reducao + '_rank_by_algorithm_' + tipo + '_' + order + '_' + str(
					alpha) + '_' + name + '_rec.csv'
			ranking_rec['ALGORITHM'] = name
			ranking_rec.to_csv(path_name, index=False)
			
			# SPE
			df_spe_col = df_spe[cols]
			df_spe_col.loc[:, delaunay_name] = df_spe_col['RANK_DELAUNAY'].values
			df_spe_col.drop(['RANK_DELAUNAY'], axis=1, inplace=True)
			ranking_spe = df_spe_col.groupby(['ALGORITHM']).mean()
			path_name = wd + reducao + '_rank_by_algorithm_' + tipo + '_' + order + '_' + str(
					alpha) + '_' + name + '_spe.csv'
			ranking_spe['ALGORITHM'] = name
			ranking_spe.to_csv(path_name, index=False)
			
			# F1
			df_f1_col = df_f1[cols]
			df_f1_col.loc[:, delaunay_name] = df_f1_col['RANK_DELAUNAY'].values
			df_f1_col.drop(['RANK_DELAUNAY'], axis=1, inplace=True)
			ranking_f1 = df_f1_col.groupby(['ALGORITHM']).mean()
			path_name = wd + reducao + '_rank_by_algorithm_' + tipo + '_' + order + '_' + str(
					alpha) + '_' + name + '_f1.csv'
			ranking_f1['ALGORITHM'] = name
			ranking_f1.to_csv(path_name, index=False)
			
			# GEO
			df_geo_col = df_geo[cols]
			df_geo_col.loc[:, delaunay_name] = df_geo_col['RANK_DELAUNAY'].values
			df_geo_col.drop(['RANK_DELAUNAY'], axis=1, inplace=True)
			ranking_geo = df_geo_col.groupby(['ALGORITHM']).mean()
			path_name = wd + reducao + '_rank_by_algorithm_' + tipo + '_' + order + '_' + str(
					alpha) + '_' + name + '_geo.csv'
			ranking_geo['ALGORITHM'] = name
			ranking_geo.to_csv(path_name, index=False)
			
			# IBA
			df_iba_col = df_iba[cols]
			df_iba_col.loc[:, delaunay_name] = df_iba_col['RANK_DELAUNAY'].values
			df_iba_col.drop(['RANK_DELAUNAY'], axis=1, inplace=True)
			ranking_iba = df_iba_col.groupby(['ALGORITHM']).mean()
			path_name = wd + reducao + '_rank_by_algorithm_' + tipo + '_' + order + '_' + str(
					alpha) + '_' + name + '_iba.csv'
			ranking_iba['ALGORITHM'] = name
			ranking_iba.to_csv(path_name, index=False)
			
			# AUC
			df_auc_col = df_auc[cols]
			df_auc_col.loc[:, delaunay_name] = df_auc_col['RANK_DELAUNAY'].values
			df_auc_col.drop(['RANK_DELAUNAY'], axis=1, inplace=True)
			ranking_auc = df_auc_col.groupby(['ALGORITHM']).mean()
			path_name = wd + reducao + '_rank_by_algorithm_' + tipo + '_' + order + '_' + str(
					alpha) + '_' + name + '_auc.csv'
			ranking_auc['ALGORITHM'] = name
			ranking_auc.to_csv(path_name, index=False)
	
	def rank_by_algorithm_dataset(self, filename):
		
		df = pd.read_csv(filename)
		df_temp = df.groupby(by=['ALGORITHM'])
		for name, group in df_temp:
			group = group.reset_index()
			group.drop('index', axis=1, inplace=True)
			df_temp1 = group.groupby(by=['DATASET'])
			for name1, group1 in df_temp1:
				group1 = group1.reset_index()
				group1.drop('index', axis=1, inplace=True)
				group1['rank_f1'] = group1['F1'].rank(ascending=False)
				group1['rank_geo'] = group1['GEO'].rank(ascending=False)
				group1['rank_iba'] = group1['IBA'].rank(ascending=False)
				group1['rank_auc'] = group1['AUC'].rank(ascending=False)
				group1.to_csv('./../output_dir/rank/rank_algorithm_dataset_' + name + '_' + name1 + '.csv', index=False)
	
	def rank_by_algorithm_dataset_only_dto(self, filename):
		
		df = pd.read_csv(filename)
		df = df[df['PREPROC'] != '_SMOTE']
		df = df[df['PREPROC'] != '_Geometric_SMOTE']
		df = df[df['PREPROC'] != '_Borderline1']
		df = df[df['PREPROC'] != '_Borderline2']
		df = df[df['PREPROC'] != '_smoteSVM']
		df = df[df['PREPROC'] != '_train']
		
		df_temp = df.groupby(by=['ALGORITHM'])
		for name, group in df_temp:
			group = group.reset_index()
			group.drop('index', axis=1, inplace=True)
			df_temp1 = group.groupby(by=['DATASET'])
			for name1, group1 in df_temp1:
				group1 = group1.reset_index()
				group1.drop('index', axis=1, inplace=True)
				group1['rank_f1'] = group1['F1'].rank(ascending=False)
				group1['rank_geo'] = group1['GEO'].rank(ascending=False)
				group1['rank_iba'] = group1['IBA'].rank(ascending=False)
				group1['rank_auc'] = group1['AUC'].rank(ascending=False)
				group1.to_csv(
						'./../output_dir/rank/only_dto/rank_algorithm_dataset_only_dto_' + name + '_' + name1 + '.csv',
						index=False)
				
				df_graph = group1.copy()
				df_graph = df_graph.replace('area', 1)
				df_graph = df_graph.replace('volume', 2)
				df_graph = df_graph.replace('area_volume_ratio', 3)
				df_graph = df_graph.replace('edge_ratio', 4)
				df_graph = df_graph.replace('radius_ratio', 5)
				df_graph = df_graph.replace('aspect_ratio', 6)
				df_graph = df_graph.replace('max_solid_angle', 7)
				df_graph = df_graph.replace('min_solid_angle', 8)
				df_graph = df_graph.replace('solid_angle', 9)
				
				legend = ['area', 'volume', 'area_volume_ratio', 'edge_ratio', 'radius_ratio', 'aspect_ratio',
				          'max_solid_angle', 'min_solid_angle', 'solid_angle']
				
				x = df_graph['ORDER'].values
				y = df_graph['ALPHA'].values.astype(float)
				dz = df_graph['AUC'].values
				
				N = x.shape[0]
				z = np.zeros(N)
				dx = 0.2 * np.ones(N)
				dy = 0.2 * np.ones(N)
				
				fig = plt.figure(figsize=(12, 8))
				ax1 = fig.add_subplot(111, projection='3d')
				cs = ['r', 'g', 'b'] * 9
				ax1.bar3d(x, y, z, dx, dy, dz, color=cs)
				
				ax1.set_ylabel('Alpha')
				ax1.set_xlabel('\n\n\n\n\nGeometry')
				ax1.set_zlabel('AUC')
				ax1.set_title('Geometry x Alpha \n Algorithm = ' + name + '\n Dataset = ' + name1)
				ax1.set_xticklabels(legend)
				ax1.legend()
				plt.show()
				
				fig = plt.figure(figsize=(12, 8))
				ax = Axes3D(fig)
				surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.5)
				fig.colorbar(surf, shrink=0.5, aspect=7)
				ax.set_xlabel('Alpha')
				ax.set_ylabel('\n\n\n\n\nGeometry')
				ax.set_zlabel('AUC')
				ax.set_title('Geometry x Alpha \n Algorithm = ' + name + '\n Dataset = ' + name1)
				ax.set_yticklabels(legend)
				ax.legend()
				plt.savefig('./../output_dir/rank/only_dto/only_dto_geometry_by_alpha_' + name + '_' + name1 + '.pdf')
				plt.show()
	
	def rank_by_measures_only_dto(self, filename):
		
		best_geometry = pd.DataFrame(columns=['PREPROC', 'M', 'ALGORITHM', 'MEDIA_RANK'])
		df = pd.read_csv(filename)
		df = df[df['PREPROC'] != '_SMOTE']
		df = df[df['PREPROC'] != '_Geometric_SMOTE']
		df = df[df['PREPROC'] != '_Borderline1']
		df = df[df['PREPROC'] != '_Borderline2']
		df = df[df['PREPROC'] != '_smoteSVM']
		df = df[df['PREPROC'] != '_train']
		i = 0
		df_temp = df.groupby(by=['ALGORITHM'])
		for name, group in df_temp:
			group = group.reset_index()
			group.drop('index', axis=1, inplace=True)
			group['rank_f1'] = group['F1'].rank(ascending=False)
			group['rank_geo'] = group['GEO'].rank(ascending=False)
			group['rank_iba'] = group['IBA'].rank(ascending=False)
			group['rank_auc'] = group['AUC'].rank(ascending=False)
			
			# AUC
			group = group.sort_values(by=['rank_auc'])
			media_rank_auc = group.groupby('PREPROC')['rank_auc'].mean()
			df_media_rank_auc = pd.DataFrame(columns=['PREPROC', 'MEDIA_RANK_AUC'])
			df_media_rank_auc['PREPROC'] = media_rank_auc.index
			df_media_rank_auc['MEDIA_RANK_AUC'] = media_rank_auc.values
			df_media_rank_auc.sort_values(by=['MEDIA_RANK_AUC'], ascending=True, inplace=True)
			df_media_rank_auc.reset_index(inplace=True)
			df_media_rank_auc.drop('index', axis=1, inplace=True)
			best_auc_geometry = df_media_rank_auc.loc[0]
			
			# GEO
			group = group.sort_values(by=['rank_geo'])
			media_rank_geo = group.groupby('PREPROC')['rank_geo'].mean()
			df_media_rank_geo = pd.DataFrame(columns=['PREPROC', 'MEDIA_RANK_GEO'])
			df_media_rank_geo['PREPROC'] = media_rank_geo.index
			df_media_rank_geo['MEDIA_RANK_GEO'] = media_rank_geo.values
			df_media_rank_geo.sort_values(by=['MEDIA_RANK_GEO'], ascending=True, inplace=True)
			df_media_rank_geo.reset_index(inplace=True)
			df_media_rank_geo.drop('index', axis=1, inplace=True)
			best_geo_geometry = df_media_rank_geo.loc[0]
			
			# IBA
			group = group.sort_values(by=['rank_iba'])
			media_rank_iba = group.groupby('PREPROC')['rank_iba'].mean()
			df_media_rank_iba = pd.DataFrame(columns=['PREPROC', 'MEDIA_RANK_IBA'])
			df_media_rank_iba['PREPROC'] = media_rank_iba.index
			df_media_rank_iba['MEDIA_RANK_IBA'] = media_rank_iba.values
			df_media_rank_iba.sort_values(by=['MEDIA_RANK_IBA'], ascending=True, inplace=True)
			df_media_rank_iba.reset_index(inplace=True)
			df_media_rank_iba.drop('index', axis=1, inplace=True)
			best_iba_geometry = df_media_rank_iba.loc[0]
			
			# F1
			group = group.sort_values(by=['rank_f1'])
			media_rank_f1 = group.groupby('PREPROC')['rank_f1'].mean()
			df_media_rank_f1 = pd.DataFrame(columns=['PREPROC', 'MEDIA_RANK_F1'])
			df_media_rank_f1['PREPROC'] = media_rank_f1.index
			df_media_rank_f1['MEDIA_RANK_F1'] = media_rank_f1.values
			df_media_rank_f1.sort_values(by=['MEDIA_RANK_F1'], ascending=True, inplace=True)
			df_media_rank_f1.reset_index(inplace=True)
			df_media_rank_f1.drop('index', axis=1, inplace=True)
			best_f1_geometry = df_media_rank_f1.loc[0]
			
			best_geometry.loc[i + 0, 'PREPROC'] = best_auc_geometry[0]
			best_geometry.loc[i + 0, 'MEDIA_RANK'] = best_auc_geometry[1]
			best_geometry.loc[i + 0, 'ALGORITHM'] = name
			best_geometry.loc[i + 0, 'M'] = 'AUC'
			best_geometry.loc[i + 1, 'PREPROC'] = best_geo_geometry[0]
			best_geometry.loc[i + 1, 'MEDIA_RANK'] = best_geo_geometry[1]
			best_geometry.loc[i + 1, 'ALGORITHM'] = name
			best_geometry.loc[i + 1, 'M'] = 'GEO'
			best_geometry.loc[i + 2, 'PREPROC'] = best_iba_geometry[0]
			best_geometry.loc[i + 2, 'MEDIA_RANK'] = best_iba_geometry[1]
			best_geometry.loc[i + 2, 'ALGORITHM'] = name
			best_geometry.loc[i + 2, 'M'] = 'IBA'
			best_geometry.loc[i + 3, 'PREPROC'] = best_f1_geometry[0]
			best_geometry.loc[i + 3, 'MEDIA_RANK'] = best_f1_geometry[1]
			best_geometry.loc[i + 3, 'ALGORITHM'] = name
			best_geometry.loc[i + 3, 'M'] = 'F1'
			i += 4
			group.to_csv('./../output_dir/rank/rank_by_measures' + '_' + name + '.csv', index=False)
		
		best_geometry.to_csv('./../output_dir/rank/best_dto_geometry_rank.csv', index=False)
	
	def find_best_dto(self):
		'''
		Find best DTO geometry and alpha parameter
		:return:
		'''
		
		df = pd.read_csv('./../output_dir/rank/rank_by_measures.csv')
		# AUC
		best_dto_auc = df.groupby(['ORDER', 'ALPHA', 'ALGORITHM'])['rank_auc']
		short = best_dto_auc.min().sort_values()
		min_auc_rank = short[0]
		df_min_auc = df[df['rank_auc'] == min_auc_rank]
		number = Counter(df_min_auc['PREPROC'])
		auc_choices = number.most_common()
		# GEO
		best_dto_geo = df.groupby(['ORDER', 'ALPHA', 'ALGORITHM'])['rank_geo']
		short = best_dto_geo.min().sort_values()
		min_geo_rank = short[0]
		df_min_geo = df[df['rank_geo'] == min_geo_rank]
		number = Counter(df_min_geo['PREPROC'])
		geo_choices = number.most_common()
		
		# IBA
		best_dto_iba = df.groupby(['ORDER', 'ALPHA', 'ALGORITHM'])['rank_iba']
		short = best_dto_iba.min().sort_values()
		min_iba_rank = short[0]
		df_min_iba = df[df['rank_iba'] == min_iba_rank]
		number = Counter(df_min_iba['PREPROC'])
		iba_choices = number.most_common()
		
		# F1
		best_dto_f1 = df.groupby(['ORDER', 'ALPHA', 'ALGORITHM'])['rank_f1']
		short = best_dto_f1.min().sort_values()
		min_f1_rank = short[0]
		df_min_f1 = df[df['rank_f1'] == min_f1_rank]
		number = Counter(df_min_f1['PREPROC'])
		f1_choices = number.most_common()
		
		d1 = {}
		d2 = {}
		d3 = {}
		d4 = {}
		d1.update(auc_choices)
		d2.update(geo_choices)
		d3.update(iba_choices)
		d4.update(f1_choices)
		
		print(auc_choices)
		print(geo_choices)
		print(iba_choices)
		print(f1_choices)
		
		x = np.arange(len(auc_choices))
		fig = plt.figure(figsize=(12, 12))
		fig.autofmt_xdate(rotation=90, ha='center')
		x1 = np.arange(len(geo_choices))
		ax = plt.subplot(111)
		ax.bar(x1, d1.values(), width=0.2, color='b', align='center')
		ax.bar(x1 - 0.2, d2.values(), width=0.2, color='g', align='center')
		ax.bar(x1 - 0.4, d3.values(), width=0.2, color='r', align='center')
		ax.bar(x1 - 0.6, d4.values(), width=0.2, color='c', align='center')
		ax.legend(('AUC', 'GEO', 'IBA', 'F1'))
		plt.xticks(x1, d1.keys(), rotation=90)
		plt.title("Best DTO", fontsize=17)
		plt.ylabel('FREQUENCY WON')
		
		ax.grid(which='both')
		
		plt.savefig('./../output_dir/rank/best_dto_geometry_alpha.pdf', dpi=200)
		# plt.show()
		plt.close()
	
	def graficos(self):
		order = ['area', 'volume', 'area_volume_ratio', 'edge_ratio', 'radius_ratio', 'aspect_ratio', 'max_solid_angle',
		         'min_solid_angle', 'solid_angle']
		alpha = [1, 4, 9]
		algorithm = ['RF', 'KNN', 'DTREE', 'GNB', 'LRG', 'ABC', 'MLP', 'QDA', 'SVM', 'SGD']
		pref = 'pca_total_rank_biclasse_'
		measures = ['auc', 'geo', 'iba', 'f1']
		preproc = ["_train", "_SMOTE", "_Borderline1", "_Borderline2", "_smoteSVM", "_Geometric_SMOTE"]
		
		dfrank = pd.DataFrame(columns=['ALGORITHM', 'UNIT', 'PREPROC', 'ALPHA', 'MEDIA_RANK_ORIGINAL', 'MEDIA_RANK_DTO',
		                               'MEDIA_RANK_GEO_SMOTE',
		                               'MEDIA_RANK_SMOTE', 'MEDIA_RANK_SMOTE_SVM', 'MEDIA_RANK_B1', 'MEDIA_RANK_B2'])
		i = 0
		for m in measures:
			for o in order:
				for a in alpha:
					for alg in algorithm:
						df = pd.read_csv(
								'./../rank/pca_biclasse/' + pref + o + '_' + str(a) + '_' + alg + '_' + m + '.csv')
						dfrank.loc[i, 'ALGORITHM'] = alg
						dfrank.loc[i, 'UNIT'] = m
						dfrank.loc[i, 'PREPROC'] = o
						dfrank.loc[i, 'ALPHA'] = a
						
						mro = df.RANK_ORIGINAL.mean()
						mrdto = df.RANK_DELAUNAY.mean()
						mrgeosmote = df.RANK_GEOMETRIC_SMOTE.mean()
						mrs = df.RANK_SMOTE.mean()
						mrssvm = df.RANK_SMOTE_SVM.mean()
						mrbl1 = df.RANK_BORDERLINE1.mean()
						mrbl2 = df.RANK_BORDERLINE2.mean()
						
						dfrank.loc[i, 'MEDIA_RANK_ORIGINAL'] = mro
						dfrank.loc[i, 'MEDIA_RANK_DTO'] = mrdto
						dfrank.loc[i, 'MEDIA_RANK_GEO_SMOTE'] = mrgeosmote
						dfrank.loc[i, 'MEDIA_RANK_SMOTE'] = mrs
						dfrank.loc[i, 'MEDIA_RANK_SMOTE_SVM'] = mrssvm
						dfrank.loc[i, 'MEDIA_RANK_B1'] = mrbl1
						dfrank.loc[i, 'MEDIA_RANK_B2'] = mrbl2
						i += 1
		
		dfrank.to_csv('./../output_dir/media_rank_all_alpha_order.csv', index=False)
	
	def rank_dto_by(self,geometry):
		M = ['_pre.csv', '_rec.csv', '_spe.csv', '_f1.csv', '_geo.csv','_iba.csv', '_auc.csv']
		#M = ['_pre.csv', '_rec.csv', '_spe.csv', '_f1.csv', '_geo.csv', '_iba.csv']
		
		df_media_rank = pd.DataFrame(columns=['ALGORITHM', 'RANK_ORIGINAL', 'RANK_SMOTE',
		                                  'RANK_SMOTE_SVM', 'RANK_BORDERLINE1', 'RANK_BORDERLINE2',
		                                  'RANK_GEOMETRIC_SMOTE', 'RANK_DELAUNAY','unit'])
		
		#name = './../output_dir/biclass/total_rank/pca_total_rank_biclasse_'+geometry + '_'
		name = './../output_dir/multiclass/total_rank/pca_total_rank_multiclasse_' + geometry + '_'
		for m in M:
			i = 0
			for c in classifiers_list:
				df = pd.read_csv(name + c + m)
				rank_original = df.RANK_ORIGINAL.mean()
				rank_smote = df.RANK_SMOTE.mean()
				rank_smote_svm = df.RANK_SMOTE_SVM.mean()
				rank_b1 = df.RANK_BORDERLINE1.mean()
				rank_b2 = df.RANK_BORDERLINE2.mean()
				rank_geo_smote = df.RANK_GEOMETRIC_SMOTE.mean()
				rank_dto = df.RANK_DELAUNAY.mean()
				df_media_rank.loc[i,'ALGORITHM'] = df.loc[0,'ALGORITHM']
				df_media_rank.loc[i,'RANK_ORIGINAL'] = rank_original
				df_media_rank.loc[i, 'RANK_SMOTE'] = rank_smote
				df_media_rank.loc[i,'RANK_SMOTE_SVM'] = rank_smote_svm
				df_media_rank.loc[i,'RANK_BORDERLINE1'] = rank_b1
				df_media_rank.loc[i, 'RANK_BORDERLINE2'] = rank_b2
				df_media_rank.loc[i,'RANK_GEOMETRIC_SMOTE'] = rank_geo_smote
				df_media_rank.loc[i,'RANK_DELAUNAY'] = rank_dto
				df_media_rank.loc[i,'unit'] = df.loc[0,'unit']
				i += 1
			
			dfmediarank = df_media_rank.copy()
			dfmediarank = dfmediarank.sort_values('RANK_DELAUNAY')
			
			dfmediarank.loc[i,'ALGORITHM'] = 'avarage'
			dfmediarank.loc[i, 'RANK_ORIGINAL'] = df_media_rank['RANK_ORIGINAL'].mean()
			dfmediarank.loc[i, 'RANK_SMOTE'] = df_media_rank['RANK_SMOTE'].mean()
			dfmediarank.loc[i, 'RANK_SMOTE_SVM'] = df_media_rank['RANK_SMOTE_SVM'].mean()
			dfmediarank.loc[i, 'RANK_BORDERLINE1'] = df_media_rank['RANK_BORDERLINE1'].mean()
			dfmediarank.loc[i, 'RANK_BORDERLINE2'] = df_media_rank['RANK_BORDERLINE2'].mean()
			dfmediarank.loc[i, 'RANK_GEOMETRIC_SMOTE'] = df_media_rank['RANK_GEOMETRIC_SMOTE'].mean()
			dfmediarank.loc[i, 'RANK_DELAUNAY'] = df_media_rank['RANK_DELAUNAY'].mean()
			dfmediarank.loc[i, 'unit'] = df.loc[0, 'unit']
			i += 1
			dfmediarank.loc[i, 'ALGORITHM'] = 'std'
			dfmediarank.loc[i, 'RANK_ORIGINAL'] = df_media_rank['RANK_ORIGINAL'].std()
			dfmediarank.loc[i, 'RANK_SMOTE'] = df_media_rank['RANK_SMOTE'].std()
			dfmediarank.loc[i, 'RANK_SMOTE_SVM'] = df_media_rank['RANK_SMOTE_SVM'].std()
			dfmediarank.loc[i, 'RANK_BORDERLINE1'] = df_media_rank['RANK_BORDERLINE1'].std()
			dfmediarank.loc[i, 'RANK_BORDERLINE2'] = df_media_rank['RANK_BORDERLINE2'].std()
			dfmediarank.loc[i, 'RANK_GEOMETRIC_SMOTE'] = df_media_rank['RANK_GEOMETRIC_SMOTE'].std()
			dfmediarank.loc[i, 'RANK_DELAUNAY'] = df_media_rank['RANK_DELAUNAY'].std()
			dfmediarank.loc[i, 'unit'] = df.loc[0, 'unit']
			#dfmediarank.to_csv('./../output_dir/results_media_rank_biclass_' + geometry + m,index=False)
			dfmediarank.to_csv('./../output_dir/results_media_rank_multiclass_'+geometry + m, index=False)
		
	def grafico_variacao_alpha(self):
		M = ['_geo', '_iba']
		order = ['solid_angle','min_solid_angle','max_solid_angle']
		
		# Dirichlet Distribution alphas
		alphas = np.arange(1, 10, 0.5)
		
		df_alpha_variations_rank = pd.DataFrame()
		df_alpha_variations_rank['alphas'] = alphas
		df_alpha_variations_rank.index=alphas
		
		for m in M:
			for o in order:
				for a in alphas:
					filename = './../variacao_alpha/' + o + m +'/results_media_rank_multiclass_'+o +'_'+str(a)+ m + '.csv'
					print(filename)
					df = pd.read_csv(filename)
					mean = df.loc[10,'RANK_DELAUNAY']
					df_alpha_variations_rank.loc[a,'AVARAGE_RANK'] = mean
				if m == '_geo':
					measure = 'GEO'
				if m == '_iba':
					measure = 'IBA'
				if m == '_auc':
					measure = 'AUC'
				
				fig, ax = plt.subplots()
				ax.set_title('DTO AVARAGE RANK\n ' +'GEOMETRY = '+ o +'\nMEASURE = '+ measure,fontsize=10)
				ax.set_xlabel('Alpha')
				ax.set_ylabel('Rank')
				plt.ylabel('Rank')
				plt.xlabel('Alpha')
				plt.title('DTO AVARAGE RANK\n ' +'GEOMETRY = '+ o +'\nMEASURE = '+ measure,fontsize=10)
				
				ax.plot(df_alpha_variations_rank['AVARAGE_RANK'],marker='d')
				legend = ax.legend(loc='upper right', shadow=True, fontsize='small')
				# Put a nicer background color on the legend.
				legend.get_frame().set_facecolor('C0')
				fig.savefig('./../variacao_alpha/graphics/'+o + '_'+measure+'.png',dpi=125)
				plt.show()
				plt.close()