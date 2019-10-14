from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np
import scikit_posthocs as sp
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
import time
import Orange

from datasetsDelaunay import dataset_list_bi, dataset_list_mult
from folders import dir_melhor_pca_biclasse, dir_melhor_pca_multiclasse, dir_melhor_isomap_biclasse, \
	dir_melhor_isomap_multiclasse, dir_melhor_isomap_pca, pca_biclasse, pca_multiclasse, isomap_biclasse, \
	isomap_multiclasse, work_delaunay_dir, output_dir
from parameters import delaunay_preproc_type, classifiers, metricas_multiclasse, classifiers_multiclasse, \
	delaunay_preproc_type_multiclasse, metricas_biclasse, delaunay_preproc_variados_type_multiclasse, preproc_type, \
	delaunay_multiclasse_corrigido, order, alphas


class Statistics:
	
	def __init__(self):
		pass
	
	def melhor_pca_biclasse(self):
		for method in delaunay_preproc_type:
			for mbiclasse in metricas_biclasse:
				lista = []
				for alg in classifiers:
					filename_pca_biclass = 'pca_total_rank_biclasse'
					filename_pca_biclass = filename_pca_biclass + method + alg + mbiclasse + '.csv'
					df = pd.read_csv(pca_biclasse + filename_pca_biclass)
					lista.append(df)
				# concatena
				df = lista.pop()
				for i in np.arange(0, len(lista)):
					df = pd.concat([df, lista[i]])
				df.to_csv(dir_melhor_pca_biclasse + 'melhor_pca_biclasse_' + mbiclasse + '.csv')
				# calcula rank medio
				df = df[['RANK_ORIGINAL', 'RANK_SMOTE', 'RANK_SMOTE_SVM', 'RANK_BORDERLINE1',
				         'RANK_BORDERLINE2','RANK_GEOSMOTE', 'RANK_DELAUNAY']]
				media_rank = df.mean(axis=0)
				# grafico CD
				identificadores = ['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', method]
				avranks = list(media_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(dataset_list_bi))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						dir_melhor_pca_biclasse + 'figurasCD/' + 'cd_' + 'pca_' + 'biclasse' + method + '_' + mbiclasse + '.pdf')
				plt.close()
	
	def melhor_pca_multiclasse(self):
		for method in delaunay_preproc_type_multiclasse:
			for mbiclasse in metricas_multiclasse:
				lista = []
				for alg in classifiers_multiclasse:
					filename_pca_multiclasse = 'pca_total_rank_multiclasse'
					filename_pca_multiclasse = filename_pca_multiclasse + method + alg + mbiclasse + '.csv'
					df = pd.read_csv(pca_multiclasse + filename_pca_multiclasse)
					lista.append(df)
				# concatena
				df = lista.pop()
				for i in np.arange(0, len(lista)):
					df = pd.concat([df, lista[i]])
				df.to_csv(dir_melhor_pca_multiclasse + 'melhor_pca_multiclasse_' + mbiclasse + '.csv')
				# calcula rank medio
				df = df[['RANK_ORIGINAL', 'RANK_SMOTE', 'RANK_SMOTE_SVM', 'RANK_BORDERLINE1',
				         'RANK_BORDERLINE2','RANK_GEOSMOTE', 'RANK_DELAUNAY']]
				media_rank = df.mean(axis=0)
				# grafico CD
				identificadores = ['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', method]
				avranks = list(media_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(dataset_list_bi))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						dir_melhor_pca_multiclasse + 'figurasCD/' + 'cd_' + 'pca_' + 'multiclasse' + method + '_' + mbiclasse + '.pdf')
				plt.close()
	
	def melhor_isomap_biclasse(self):
		for method in delaunay_preproc_type:
			for mbiclasse in metricas_biclasse:
				lista = []
				for alg in classifiers:
					filename_isomap_biclass = 'isomap_total_rank_biclasse'
					filename_isomap_biclass = filename_isomap_biclass + method + alg + mbiclasse + '.csv'
					df = pd.read_csv(isomap_biclasse + filename_isomap_biclass)
					lista.append(df)
				# concatena
				df = lista.pop()
				for i in np.arange(0, len(lista)):
					df = pd.concat([df, lista[i]])
				df.to_csv(dir_melhor_isomap_biclasse + 'melhor_isomap_biclasse_' + mbiclasse + '.csv')
				# calcula rank medio
				df = df[['RANK_ORIGINAL', 'RANK_SMOTE', 'RANK_SMOTE_SVM', 'RANK_BORDERLINE1',
				         'RANK_BORDERLINE2', 'RANK_GEOSMOTE','RANK_DELAUNAY']]
				media_rank = df.mean(axis=0)
				# grafico CD
				identificadores = ['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', method]
				avranks = list(media_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(dataset_list_bi))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						dir_melhor_isomap_biclasse + 'figurasCD/' + 'cd_' + 'isomap_' + 'biclasse' + method + '_' + mbiclasse + '.pdf')
				plt.close()
	
	def melhor_isomap_multiclasse(self):
		for method in delaunay_preproc_type_multiclasse:
			for mbiclasse in metricas_multiclasse:
				lista = []
				for alg in classifiers_multiclasse:
					filename_isomap_multiclasse = 'isomap_total_rank_multiclasse'
					filename_isomap_multiclasse = filename_isomap_multiclasse + method + alg + mbiclasse + '.csv'
					df = pd.read_csv(isomap_multiclasse + filename_isomap_multiclasse)
					lista.append(df)
				# concatena
				df = lista.pop()
				for i in np.arange(0, len(lista)):
					df = pd.concat([df, lista[i]])
				df.to_csv(dir_melhor_isomap_multiclasse + 'melhor_isomap_multiclasse_' + mbiclasse + '.csv')
				# calcula rank medio
				df = df[['RANK_ORIGINAL', 'RANK_SMOTE', 'RANK_SMOTE_SVM', 'RANK_BORDERLINE1',
				         'RANK_BORDERLINE2', 'RANK_GEOSMOTE', 'RANK_DELAUNAY']]
				media_rank = df.mean(axis=0)
				# grafico CD
				identificadores = ['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', method]
				avranks = list(media_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(dataset_list_bi))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						dir_melhor_isomap_multiclasse + 'figurasCD/' + 'cd_' + 'isomap_' + 'multiclasse' + method + '_' + mbiclasse + '.pdf')
				plt.close()
	
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
	
	def melhor_isomap_pca_multiclasse(self):
		for mbiclasse in metricas_multiclasse:
			# ISOMAP
			lista_df_aux = []
			for method in delaunay_preproc_type_multiclasse:
				lista = []
				for alg in classifiers_multiclasse:
					filename_isomap_multiclasse = 'isomap_total_rank_multiclasse'
					filename_isomap_multiclasse = filename_isomap_multiclasse + method + alg + mbiclasse + '.csv'
					df = pd.read_csv(isomap_multiclasse + filename_isomap_multiclasse)
					lista.append(df)
				# concatena
				df = lista.pop()
				for i in np.arange(0, len(lista)):
					df = pd.concat([df, lista[i]], ignore_index=True)
				df['ZIP'] = 'ISOMAP'
				lista_df_aux.append(df)
			# PCA
			for method in delaunay_preproc_type_multiclasse:
				lista = []
				for alg in classifiers_multiclasse:
					filename_pca_multiclasse = 'pca_total_rank_multiclasse'
					filename_pca_multiclasse = filename_pca_multiclasse + method + alg + mbiclasse + '.csv'
					df = pd.read_csv(pca_multiclasse + filename_pca_multiclasse)
					lista.append(df)
				# concatena
				df = lista.pop()
				for i in np.arange(0, len(lista)):
					df = pd.concat([df, lista[i]], ignore_index=True)
				df['ZIP'] = 'PCA'
				lista_df_aux.append(df)
			
			df = lista_df_aux.pop()
			for i in np.arange(0, len(lista_df_aux)):
				df = pd.concat([df, lista_df_aux[i]], ignore_index=True)
			
			df.to_csv(dir_melhor_isomap_pca + 'melhor_isomap_pca_multiclasse_' + mbiclasse + '.csv')
		
		v = len(classifiers_multiclasse) * len(dataset_list_mult)
		idx = np.arange(0, v + 10)
		col_pca = [i + str('_PCA') for i in delaunay_multiclasse_corrigido]
		col_isomap = [i + str('_ISOMAP') for i in delaunay_multiclasse_corrigido]
		df_tabela = pd.DataFrame(
				columns=['DATASET', 'ALGORITHM', 'ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1',
				         'BORDERLINE2'] + col_isomap + col_pca, index=idx)
		
		for mbiclasse in metricas_multiclasse:
			df = pd.read_csv(dir_melhor_isomap_pca + 'melhor_isomap_pca_multiclasse_' + mbiclasse + '.csv')
			df_temp = df.groupby(by=['DATASET', 'ALGORITHM'])
			i = idx[0]
			for name, group in df_temp:
				group = group.reset_index()
				df_tabela.at[i, 'DATASET'] = group.loc[0, 'DATASET']
				df_tabela.at[i, 'ALGORITHM'] = group.loc[0, 'ALGORITHM']
				df_tabela.at[i, 'ORIGINAL'] = group.loc[0, 'ORIGINAL']
				df_tabela.at[i, 'SMOTE'] = group.loc[0, 'SMOTE']
				df_tabela.at[i, 'SMOTE_SVM'] = group.loc[0, 'SMOTE_SVM']
				df_tabela.at[i, 'BORDERLINE1'] = group.loc[0, 'BORDERLINE1']
				df_tabela.at[i, 'BORDERLINE2'] = group.loc[0, 'BORDERLINE2']
				df_d = group.groupby(by=['DELAUNAY', 'delaunayTYPE', 'ZIP'])
				
				# for n, g in df_d:
				#    delaunay = n[1].replace('delaunay','delaunay')
				#    delaunay = delaunay + '_'+n[2]
				#    df_tabela.at[i,delaunay] = n[0]
				
				df_tabela.dropna(inplace=True)
				i = i + 1
				print(i)
			
			df_tabela_rank = df_tabela.loc[:, 'ORIGINAL':'_delaunay_solid_angle_9_PCA']
			# calcula rank linha a linha
			rank = df_tabela_rank.rank(axis=1, ascending=False)
			# calcula rank medio
			media_rank = df_tabela_rank.mean(axis=0)
			# grafico CD
			identificadores = media_rank.index.values
			avranks = list(media_rank)
			cd = self.compute_CD_customizado(avranks, df_tabela.shape[0])
			Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
			plt.savefig(
					dir_melhor_isomap_pca + 'figurasCD/' + 'cd_' + 'isomap_pca_' + 'multiclasse_' + mbiclasse + '.pdf')
			plt.close()
	
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
	
	def calcula_media_folds_multiclasse(self, df):
		t = pd.Series(data=np.arange(0, df.shape[0], 1))
		dfr = pd.DataFrame(
				columns=['MODE', 'DATASET', 'PREPROC', 'ALGORITHM', 'PRE', 'REC', 'SPE', 'F1', 'GEO', 'IBA'],
				index=np.arange(0, int(t.shape[0] / 5)))
		
		df_temp = df.groupby(by=['MODE', 'DATASET', 'PREPROC', 'ALGORITHM'])
		idx = dfr.index.values
		i = idx[0]
		for name, group in df_temp:
			group = group.reset_index()
			dfr.at[i, 'MODE'] = group.loc[0, 'MODE']
			dfr.at[i, 'DATASET'] = group.loc[0, 'DATASET']
			dfr.at[i, 'PREPROC'] = group.loc[0, 'PREPROC']
			dfr.at[i, 'ALGORITHM'] = group.loc[0, 'ALGORITHM']
			dfr.at[i, 'PRE'] = group['PRE'].mean()
			dfr.at[i, 'REC'] = group['REC'].mean()
			dfr.at[i, 'SPE'] = group['SPE'].mean()
			dfr.at[i, 'F1'] = group['F1'].mean()
			dfr.at[i, 'GEO'] = group['GEO'].mean()
			dfr.at[i, 'IBA'] = group['IBA'].mean()
			i = i + 1
			print(i)
		
		dfr.to_csv('./../results/multiclasse/media_resultados_variados_multiclasse_pca.csv', index=False)
	
	def separa_pca_isomap(self, df):
		df_pca = df[df['MODE'] == 'PCA']
		df_pca = df_pca.reset_index()
		df_isomap = df[df['MODE'] == 'Isomap']
		df_isomap = df_isomap.reset_index()
		df_nc = df[df['MODE'] == 'NC']
		df_nc = df_nc.reset_index()
		df_pca = pd.concat([df_pca, df_nc])
		print('aqui')
		df_pca.to_csv('./../results/media/variados_multiclasse_pca_media.csv', index=False)
		df_isomap = pd.concat([df_isomap, df_nc])
		df_isomap.to_csv('./../results/media/variados_multiclasse_isomap_media.csv', index=False)
	
	def separa_delaunay_multiclass(self, filename):
		df = pd.read_csv(filename)
		list_base = []
		for p in np.arange(0, len(preproc_type)):
			list_base.append(df[(df['PREPROC'] == preproc_type[p])])
		df_base = list_base.pop(0)
		for i in np.arange(0, len(list_base)):
			df_base = pd.concat([df_base, list_base[i]], ignore_index=True)
		
		for p in np.arange(0, len(delaunay_preproc_variados_type_multiclasse)):
			dfr = df[(df['PREPROC'] == delaunay_preproc_variados_type_multiclasse[p])]
			df_file = pd.concat([df_base, dfr])
			df_file.to_csv(work_delaunay_dir + delaunay_preproc_variados_type_multiclasse[p] + '.csv', index=False)
	
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
				df_file.to_csv(work_delaunay_dir + '_' + o + '_' + str(a) + '.csv', index=False)
	
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
				         'GEOSMOTE','RANK_GEOSMOTE','DELAUNAY', 'RANK_DELAUNAY', 'DELAUNAY_TYPE', 'ALPHA', 'unit'])
		
		df_temp = df.groupby(by=['ALGORITHM'])
		for name, group in df_temp:
			group = group.reset_index()
			group.drop('index', axis=1, inplace=True)
			df.to_csv(pca_biclasse + reducao + '_' + tipo + '_' + order + '_' + str(alpha) + '.csv')
			
			j = 0
			for d in dataset_list_bi:
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
					df_tabela.at[j, 'GEOSMOTE'] = aux.at[indice, m]
					
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
			
			pre = df_pre[['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2','GEOSMOTE', 'DELAUNAY']]
			rec = df_rec[['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2','GEOSMOTE', 'DELAUNAY']]
			spe = df_spe[['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2','GEOSMOTE', 'DELAUNAY']]
			f1 = df_f1[['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2','GEOSMOTE', 'DELAUNAY']]
			geo = df_geo[['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2','GEOSMOTE', 'DELAUNAY']]
			iba = df_iba[['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2','GEOSMOTE', 'DELAUNAY']]
			auc = df_auc[['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2','GEOSMOTE', 'DELAUNAY']]
			
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
			df_pre['RANK_GEOSMOTE'] = pre_rank['GEOSMOTE']
			df_pre['RANK_DELAUNAY'] = pre_rank['DELAUNAY']
			
			df_rec = df_rec.reset_index()
			df_rec.drop('index', axis=1, inplace=True)
			df_rec['RANK_ORIGINAL'] = rec_rank['ORIGINAL']
			df_rec['RANK_SMOTE'] = rec_rank['SMOTE']
			df_rec['RANK_SMOTE_SVM'] = rec_rank['SMOTE_SVM']
			df_rec['RANK_BORDERLINE1'] = rec_rank['BORDERLINE1']
			df_rec['RANK_BORDERLINE2'] = rec_rank['BORDERLINE2']
			df_rec['RANK_GEOSMOTE'] = rec_rank['GEOSMOTE']
			df_rec['RANK_DELAUNAY'] = rec_rank['DELAUNAY']
			
			df_spe = df_spe.reset_index()
			df_spe.drop('index', axis=1, inplace=True)
			df_spe['RANK_ORIGINAL'] = spe_rank['ORIGINAL']
			df_spe['RANK_SMOTE'] = spe_rank['SMOTE']
			df_spe['RANK_SMOTE_SVM'] = spe_rank['SMOTE_SVM']
			df_spe['RANK_BORDERLINE1'] = spe_rank['BORDERLINE1']
			df_spe['RANK_BORDERLINE2'] = spe_rank['BORDERLINE2']
			df_spe['RANK_GEOSMOTE'] = spe_rank['GEOSMOTE']
			df_spe['RANK_DELAUNAY'] = spe_rank['DELAUNAY']
			
			df_f1 = df_f1.reset_index()
			df_f1.drop('index', axis=1, inplace=True)
			df_f1['RANK_ORIGINAL'] = f1_rank['ORIGINAL']
			df_f1['RANK_SMOTE'] = f1_rank['SMOTE']
			df_f1['RANK_SMOTE_SVM'] = f1_rank['SMOTE_SVM']
			df_f1['RANK_BORDERLINE1'] = f1_rank['BORDERLINE1']
			df_f1['RANK_BORDERLINE2'] = f1_rank['BORDERLINE2']
			df_f1['RANK_GEOSMOTE'] = f1_rank['GEOSMOTE']
			df_f1['RANK_DELAUNAY'] = f1_rank['DELAUNAY']
			
			df_geo = df_geo.reset_index()
			df_geo.drop('index', axis=1, inplace=True)
			df_geo['RANK_ORIGINAL'] = geo_rank['ORIGINAL']
			df_geo['RANK_SMOTE'] = geo_rank['SMOTE']
			df_geo['RANK_SMOTE_SVM'] = geo_rank['SMOTE_SVM']
			df_geo['RANK_BORDERLINE1'] = geo_rank['BORDERLINE1']
			df_geo['RANK_BORDERLINE2'] = geo_rank['BORDERLINE2']
			df_geo['RANK_GEOSMOTE'] = geo_rank['GEOSMOTE']
			df_geo['RANK_DELAUNAY'] = geo_rank['DELAUNAY']
			
			df_iba = df_iba.reset_index()
			df_iba.drop('index', axis=1, inplace=True)
			df_iba['RANK_ORIGINAL'] = iba_rank['ORIGINAL']
			df_iba['RANK_SMOTE'] = iba_rank['SMOTE']
			df_iba['RANK_SMOTE_SVM'] = iba_rank['SMOTE_SVM']
			df_iba['RANK_BORDERLINE1'] = iba_rank['BORDERLINE1']
			df_iba['RANK_BORDERLINE2'] = iba_rank['BORDERLINE2']
			df_iba['RANK_GEOSMOTE'] = iba_rank['GEOSMOTE']
			df_iba['RANK_DELAUNAY'] = iba_rank['DELAUNAY']
			
			df_auc = df_auc.reset_index()
			df_auc.drop('index', axis=1, inplace=True)
			df_auc['RANK_ORIGINAL'] = auc_rank['ORIGINAL']
			df_auc['RANK_SMOTE'] = auc_rank['SMOTE']
			df_auc['RANK_SMOTE_SVM'] = auc_rank['SMOTE_SVM']
			df_auc['RANK_BORDERLINE1'] = auc_rank['BORDERLINE1']
			df_auc['RANK_BORDERLINE2'] = auc_rank['BORDERLINE2']
			df_auc['RANK_GEOSMOTE'] = auc_rank['GEOSMOTE']
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
			df_pre.to_csv(wd + reducao + '_total_rank_' + tipo + '_' + order + '_' + str(alpha) + '_' + name + '_pre.csv',
			              index=False)
			df_rec.to_csv(wd + reducao + '_total_rank_' + tipo + '_' + order + '_' + str(alpha) + '_' + name + '_rec.csv',
			              index=False)
			df_spe.to_csv(wd + reducao + '_total_rank_' + tipo + '_' + order + '_' + str(alpha) + '_' + name + '_spe.csv',
			              index=False)
			df_f1.to_csv(wd + reducao + '_total_rank_' + tipo + '_' + order + '_' + str(alpha) + '_' + name + '_f1.csv',
			             index=False)
			df_geo.to_csv(wd + reducao + '_total_rank_' + tipo + '_' + order + '_' + str(alpha) + '_' + name + '_geo.csv',
			              index=False)
			df_iba.to_csv(wd + reducao + '_total_rank_' + tipo + '_' + order + '_' + str(alpha) + '_' + name + '_iba.csv',
			              index=False)
			df_auc.to_csv(wd + reducao + '_total_rank_' + tipo + '_' + order + '_' + str(alpha) + '_' + name + '_auc.csv',
			              index=False)
			
			media_pre_rank_file.to_csv(
				wd + reducao + '_media_rank_' + tipo + '_' + order + '_' + str(alpha) + '_' + name + '_pre.csv',
				index=False)
			media_rec_rank_file.to_csv(
				wd + reducao + '_media_rank_' + tipo + '_' + order + '_' + str(alpha) + '_' + name + '_rec.csv',
				index=False)
			media_spe_rank_file.to_csv(
				wd + reducao + '_media_rank_' + tipo + '_' + order + '_' + str(alpha) + '_' + name + '_spe.csv',
				index=False)
			media_f1_rank_file.to_csv(
				wd + reducao + '_media_rank_' + tipo + '_' + order + '_' + str(alpha) + '_' + name + '_f1.csv',
				index=False)
			media_geo_rank_file.to_csv(
				wd + reducao + '_media_rank_' + tipo + '_' + order + '_' + str(alpha) + '_' + name + '_geo.csv',
				index=False)
			media_iba_rank_file.to_csv(
				wd + reducao + '_media_rank_' + tipo + '_' + order + '_' + str(alpha) + '_' + name + '_iba.csv',
				index=False)
			media_auc_rank_file.to_csv(
				wd + reducao + '_media_rank_' + tipo + '_' + order + '_' + str(alpha) + '_' + name + '_auc.csv',
				index=False)
			
			delaunay_type = order + '_' + str(alpha)
			
			# grafico CD
			identificadores = ['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2','GEOSMOTE', 'DTO-SMOTE']
			avranks = list(media_pre_rank)
			cd = Orange.evaluation.compute_CD(avranks, len(dataset_list_bi))
			Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
			plt.savefig(wd + 'figurasCD/' + 'cd_' + reducao + '_' + tipo + '_'+ delaunay_type + '_' + name + '_pre.pdf')
			plt.close()
			
			avranks = list(media_rec_rank)
			cd = Orange.evaluation.compute_CD(avranks, len(dataset_list_bi))
			Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
			plt.savefig(wd + 'figurasCD/' + 'cd_' + reducao + '_' + tipo + '_'+ delaunay_type + '_' + name + '_rec.pdf')
			plt.close()
			
			avranks = list(media_spe_rank)
			cd = Orange.evaluation.compute_CD(avranks, len(dataset_list_bi))
			Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
			plt.savefig(wd + 'figurasCD/' + 'cd_' + reducao + '_' + tipo  + '_'+ delaunay_type +'_' + name + '_spe.pdf')
			plt.close()
			
			avranks = list(media_f1_rank)
			cd = Orange.evaluation.compute_CD(avranks, len(dataset_list_bi))
			Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
			plt.savefig(wd + 'figurasCD/' + 'cd_' + reducao + '_' + tipo  + '_'+ delaunay_type + '_' + name + '_f1.pdf')
			plt.close()
			
			avranks = list(media_geo_rank)
			cd = Orange.evaluation.compute_CD(avranks, len(dataset_list_bi))
			Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
			plt.savefig(wd + 'figurasCD/' + 'cd_' + reducao + '_' + tipo  + '_'+ delaunay_type + '_' + name + '_geo.pdf')
			plt.close()
			
			avranks = list(media_iba_rank)
			cd = Orange.evaluation.compute_CD(avranks, len(dataset_list_bi))
			Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
			plt.savefig(wd + 'figurasCD/' + 'cd_' + reducao + '_' + tipo  + '_'+ delaunay_type + '_' + name + '_iba.pdf')
			plt.close()
			
			avranks = list(media_auc_rank)
			cd = Orange.evaluation.compute_CD(avranks, len(dataset_list_bi))
			Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
			plt.savefig(wd + 'figurasCD/' + 'cd_' + reducao + '_' + tipo  + '_'+ delaunay_type + '_' + name + '_auc.pdf')
			plt.close()
			
			print('Delaunay Type= ', delaunay_type)
			print('Algorithm= ',name)
	
	def read_dir_files(self, dir_name):
		f = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
		return f
	
	def find_best_rank(self, results_dir):
		results = self.read_dir_files(results_dir)
		df = pd.DataFrame(columns=[['ARQUIVO', 'WINER']])
		i = 0
		for f in results:
			df_temp = pd.read_csv(results_dir + f)
			df.at[i, 'ARQUIVO'] = f
			df.at[i, 'WINER'] = df_temp.iloc[0, 0]
			i += 1
		
		df.to_csv(output_dir + 'best_pca_biclass_media_rank.csv')
		
		df.to_csv(output_dir + 'best_pca_biclass_media_rank.csv', index=False)
		df1 = pd.read_csv(output_dir + 'best_pca_biclass_media_rank.csv')
		df1 = df1.loc[df1['WINER'] == 'DELAUNAY']
		df1.to_csv(output_dir + 'ollyDelaunay_best_pca_biclasse_media_rank.csv', index=False)