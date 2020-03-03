# from folders import dir_isomap_biclasse
from folders import dir_pca_biclasse, output_dir
from parameters import order, alphas
from statistics import Statistics
import pandas as pd


def main():
	diag = Statistics()
	#df = pd.read_csv('./../output_dir/results_multiclass_PCA.csv')
	#diag.calcula_media_folds_multiclass(df)
	#diag.separa_delaunay_biclass('./../output_dir/resultado_media_multiclass_PCA.csv')


	# Remove others DTO from result file
	GEOMETRY = '_delaunay_area_9'
	df_best_dto = pd.read_csv('./../output_dir/resultado_media_multiclass_PCA.csv')
	df_B1 = df_best_dto[df_best_dto['PREPROC']=='_Borderline1'].copy()
	df_B2 = df_best_dto[df_best_dto['PREPROC'] == '_Borderline2'].copy()
	df_GEO = df_best_dto[df_best_dto['PREPROC'] == '_Geometric_SMOTE'].copy()
	df_SMOTE = df_best_dto[df_best_dto['PREPROC'] == '_SMOTE'].copy()
	df_SMOTEsvm = df_best_dto[df_best_dto['PREPROC'] == '_smoteSVM'].copy()
	df_original = df_best_dto[df_best_dto['PREPROC'] == '_train'].copy()
	df_dto = df_best_dto[df_best_dto['PREPROC'] == GEOMETRY].copy()
	
	df = pd.concat([df_B1,df_B2,df_GEO,df_SMOTE,df_SMOTEsvm,df_original,df_dto])
	diag.rank_by_algorithm(df, 'multiclass', './../output_dir/multiclass/', 'pca', 'area', 9)
	diag.rank_dto_by('area_9')
	
	
	#diag.rank_total_by_algorithm('biclasse', './../output_dir/results_totais/', 'pca', 'max_solid_angle', 4)
	#diag.find_best_rank('./../output_dir/media_rank/', 'best_pca_biclass_media_rank.csv')
	#diag.find_best_delaunay('./../output_dir/', 'best_pca_biclass_media_rank.csv')'''
	
	

'''for o in order:
	for a in alphas:
		df = pd.read_csv('./../output_dir/result_biclass_'+o+'_'+str(a)+'.csv')
		diag.rank_by_algorithm(df, 'biclasse', dir_pca_biclasse,'pca',o,a)
		diag.rank_total_by_algorithm('biclasse', dir_pca_biclasse,'pca',o,a)

diag.find_best_rank(dir_pca_biclasse + 'media/','best_pca_biclass_media_rank.csv')
diag.find_best_delaunay(output_dir,  'best_pca_biclass_media_rank.csv')
'''

# diag.rank_by_algorithm_dataset('./../output_dir/resultado_media_biclasse_PCA.csv')
# diag.rank_by_algorithm_dataset_only_dto('./../output_dir/resultado_media_biclasse_PCA.csv')
# diag.rank_by_measures_only_dto('./../output_dir/resultado_media_biclasse_PCA.csv')
# diag.find_best_dto()


if __name__ == '__main__':
	main()
