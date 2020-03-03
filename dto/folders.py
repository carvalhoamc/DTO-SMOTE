work_dir = './../datasets/'
folder_experiments = work_dir
output_dir = "./../output_dir/"
mesh_folder = "./../mesh/"
graph_folder = "./../graph/"
dir_pca_biclasse = './../rank/pca_biclasse/'
dir_pca_multiclasse = './../rank/pca_multiclasse/'
dir_melhor_pca_biclasse = './../rank/melhor_pca_biclasse/'
dir_melhor_pca_multiclasse ='./../rank/melhor_pca_multiclasse/'
pca_biclasse = './../rank/pca_biclasse/'
pca_multiclasse ='./../rank/pca_multiclasse/'
work_delaunay_dir = './../delaunay_files/'


train_smote_ext = ["_train", "_SMOTE", "_Borderline1", "_Borderline2", "_smoteSVM","_Geometric_SMOTE"]
preproc_type = train_smote_ext
metricas_biclasse = ['PRE', 'REC', 'SPE', 'F1', 'GEO', 'IBA', 'AUC']
metricas_multiclasse = ['PRE', 'REC', 'SPE', 'F1', 'GEO', 'IBA']

classifiers = ["_RF_", "_KNN_", "_DTREE_", "_GNB_", "_LRG_", "_ABC_", "_MLP_", "_KDA_", "_SVM_",
               "_SGD_", "_BBag_", "_EEC_", "_RUSBOOST_", "_SMOTEBOOST_"]

classifiers_multiclasse = ["_RF_", "_KNN_", "_DTREE_", "_GNB_", "_LRG_", "_ABC_", "_MLP_", "_KDA_", "_SVM_"]



'''
dir_pca_biclasse = './../rank/pca_biclasse/'
dir_isomap_biclasse = './../rank/isomap_biclasse/'
dir_isomap_multiclasse = './../rank/isomap_multiclasse/'
dir_pca_multiclasse = './../rank/pca_multiclasse/'
dir_variados_pca_multiclasse = './../rank/pca_variados_multiclasse/'
dir_variados_isomap_multiclasse = './../rank/isomap_variados_multiclasse/'

dir_melhor_pca_biclasse = './../rank/melhor_pca_biclasse/'
dir_melhor_pca_multiclasse ='./../rank/melhor_pca_multiclasse/'
dir_melhor_isomap_biclasse = './../rank/melhor_isomap_biclasse/'
dir_melhor_isomap_multiclasse = './../rank/melhor_isomap_multiclasse/'
dir_melhor_isomap_pca ='./../rank/melhor_isomap_pca/'

pca_biclasse = './../rank/pca_biclasse/'
isomap_biclasse = './../rank/isomap_biclasse/'
pca_multiclasse ='./../rank/pca_multiclasse/'
isomap_multiclasse = './../rank/isomap_multiclasse/'

work_delaunay_dir = './../delaunay_files/'
'''