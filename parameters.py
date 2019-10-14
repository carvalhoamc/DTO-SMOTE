from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

order = ['area', #nao ganhou nenhuma vez no pca biclasse
         'volume', #nao ganhou nenhuma vez no pca biclasse
         'area_volume_ratio',#ganhou 23x no pca biclasse
         'edge_ratio',#ganhou 21x no pca biclasse
         'radius_ratio',#nao ganhou nenhuma vez no pca biclasse
         'aspect_ratio',#ganhou 12x no pca biclasse
         'max_solid_angle',#ganhou 47x no pca biclasse
         'min_solid_angle',#ganhou 31x no pca biclasse
         'solid_angle']#ganhou 31x no pca biclasse

# Dirichlet Distribution alphas
alphas = [1,4,9]

# Compression Methods
projectors = [PCA()]

train_smote_ext = ["_train", "_SMOTE", "_Borderline1", "_Borderline2", "_smoteSVM","_Geometric_SMOTE"]
preproc_type = train_smote_ext
metricas_biclasse = ['PRE', 'REC', 'SPE', 'F1', 'GEO', 'IBA', 'AUC']
metricas_multiclasse = ['pre', 'rec', 'spe', 'f1', 'geo', 'iba']

classifiers = ["_RF_", "_KNN_", "_DTREE_", "_GNB_", "_LRG_", "_ABC_", "_MLP_", "_KDA_", "_SVM_",
               "_SGD_", "_BBag_", "_EEC_", "_RUSBOOST_", "_SMOTEBOOST_"]

classifiers_multiclasse = ["_RF_", "_KNN_", "_DTREE_", "_GNB_", "_LRG_", "_ABC_", "_MLP_", "_KDA_", "_SVM_"]

delaunay_preproc_type = [
	'_delaunay_max_solid_angle_1',
	'_delaunay_max_solid_angle_2',
	'_delaunay_max_solid_angle_3',
	'_delaunay_max_solid_angle_4',
	'_delaunay_max_solid_angle_5',
	'_delaunay_max_solid_angle_6',
	'_delaunay_max_solid_angle_7',
	'_delaunay_max_solid_angle_8',
	'_delaunay_max_solid_angle_9',
	'_delaunay_min_solid_angle_1',
	'_delaunay_min_solid_angle_2',
	'_delaunay_min_solid_angle_3',
	'_delaunay_min_solid_angle_4',
	'_delaunay_min_solid_angle_5',
	'_delaunay_min_solid_angle_6',
	'_delaunay_min_solid_angle_7',
	'_delaunay_min_solid_angle_8',
	'_delaunay_min_solid_angle_9',
	'_delaunay_solid_angle_1',
	'_delaunay_solid_angle_2',
	'_delaunay_solid_angle_3',
	'_delaunay_solid_angle_4',
	'_delaunay_solid_angle_5',
	'_delaunay_solid_angle_6',
	'_delaunay_solid_angle_7',
	'_delaunay_solid_angle_8',
	'_delaunay_solid_angle_9']  # 27 tipos

delaunay_biclasse_corrigido = delaunay_preproc_type

delaunay_preproc_type_multiclasse = [
	'_delaunay_max_solid_angle_1',
	'_delaunay_max_solid_angle_3',
	'_delaunay_max_solid_angle_5',
	'_delaunay_max_solid_angle_7',
	'_delaunay_max_solid_angle_9',
	'_delaunay_min_solid_angle_1',
	'_delaunay_min_solid_angle_3',
	'_delaunay_min_solid_angle_5',
	'_delaunay_min_solid_angle_7',
	'_delaunay_min_solid_angle_9',
	'_delaunay_solid_angle_1',
	'_delaunay_solid_angle_3',
	'_delaunay_solid_angle_5',
	'_delaunay_solid_angle_7',
	'_delaunay_solid_angle_9']  # 15 tipos

delaunay_multiclasse_corrigido = delaunay_preproc_type_multiclasse

delaunay_preproc_variados_type_multiclasse = ['_delaunay_area_1',
                                              '_delaunay_area_3', '_delaunay_area_5', '_delaunay_area_7',
                                              '_delaunay_area_9', '_delaunay_area_volume_ratio_1',
                                              '_delaunay_area_volume_ratio_3', '_delaunay_area_volume_ratio_5',
                                              '_delaunay_area_volume_ratio_7', '_delaunay_area_volume_ratio_9',
                                              '_delaunay_aspect_ratio_1', '_delaunay_aspect_ratio_3',
                                              '_delaunay_aspect_ratio_5', '_delaunay_aspect_ratio_7',
                                              '_delaunay_aspect_ratio_9', '_delaunay_edge_ratio_1',
                                              '_delaunay_edge_ratio_3', '_delaunay_edge_ratio_5',
                                              '_delaunay_edge_ratio_7', '_delaunay_edge_ratio_9',
                                              '_delaunay_max_solid_angle_1', '_delaunay_max_solid_angle_3',
                                              '_delaunay_max_solid_angle_5', '_delaunay_max_solid_angle_7',
                                              '_delaunay_max_solid_angle_9', '_delaunay_min_solid_angle_1',
                                              '_delaunay_min_solid_angle_3', '_delaunay_min_solid_angle_5',
                                              '_delaunay_min_solid_angle_7', '_delaunay_min_solid_angle_9',
                                              '_delaunay_radius_ratio_1', '_delaunay_radius_ratio_3',
                                              '_delaunay_radius_ratio_5', '_delaunay_radius_ratio_7',
                                              '_delaunay_radius_ratio_9', '_delaunay_solid_angle_1',
                                              '_delaunay_solid_angle_3', '_delaunay_solid_angle_5',
                                              '_delaunay_solid_angle_7', '_delaunay_solid_angle_9',
                                              '_delaunay_volume_1', '_delaunay_volume_3', '_delaunay_volume_5',
                                              '_delaunay_volume_7', '_delaunay_volume_9']
