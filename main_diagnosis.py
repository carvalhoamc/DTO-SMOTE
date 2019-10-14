from folders import dir_isomap_biclasse, dir_pca_biclasse
from parameters import order, alphas
from statistics import Statistics
import pandas as pd
import sys

sys.path.append('/home/amc/Doutorado2019/V11_PCA_BICLASS_BEST_DELAUNAY/')


def main():
	diag = Statistics()
	
	#df = pd.read_csv('./../output_dir/resultado_biclasse_PCA.csv')
	#diag.calcula_media_folds_biclasse(df)
	#diag.separa_delaunay_biclass('./../output_dir/resultado_media_biclasse_PCA.csv')
	
	#for o in order:
	#	for a in alphas:
	#		df = pd.read_csv('./../delaunay_files/_'+o+'_'+str(a)+'.csv')
	#		diag.rank_by_algorithm(df, 'biclasse', dir_pca_biclasse,'PCA',o,a)
	
	diag.find_best_rank(dir_pca_biclasse + 'txt_auc_geo_iba/media/')


if __name__ == '__main__':
	main()
	
	
	
	
	
	