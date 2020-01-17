#from folders import dir_isomap_biclasse
from folders import dir_pca_biclasse
from parameters import order, alphas
from statistics import Statistics
import pandas as pd


def main():
    diag = Statistics()
    #df = pd.read_csv('./../output_dir/results_biclass_PCA.csv')
    #diag.calcula_media_folds_biclasse(df)
    #diag.separa_delaunay_biclass('./../output_dir/resultado_media_biclasse_PCA.csv')

    for o in order:
        for a in alphas:
            #df = pd.read_csv('./../delaunay_files/_'+o+'_'+str(a)+'.csv')
            #diag.rank_by_algorithm(df, 'biclasse', dir_pca_biclasse,'pca',o,a)
            diag.rank_total_by_algorithm('biclasse', dir_pca_biclasse,'pca',o,a)



    #diag.find_best_rank(dir_pca_biclasse + 'media/','best_pca_biclass_media_rank.csv')
    #diag.find_best_delaunay(dir_pca_biclasse + 'media/', 'only_delaunay_best_pca_biclass_media_rank.csv')


if __name__ == '__main__':
    main()