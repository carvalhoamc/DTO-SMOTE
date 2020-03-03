#from folders import dir_isomap_biclasse
from folders import dir_pca_biclasse, output_dir
from parameters import order, alphas
from statistics import Statistics
import pandas as pd


def main():
    diag = Statistics()
    #df = pd.read_csv('./../output_dir/results_biclass_PCA.csv')
    #diag.calcula_media_folds_biclasse(df)
    #diag.separa_delaunay_biclass('./../output_dir/resultado_media_biclasse_PCA.csv')
    
    #Remove others DTO from result file
    
    '''df_best_dto = pd.read_csv('./../output_dir/resultado_media_multiclasse_PCA.csv')
    df_B1 = df_best_dto[df_best_dto['PREPROC']=='_Borderline1'].copy()
    df_B2 = df_best_dto[df_best_dto['PREPROC'] == '_Borderline2'].copy()
    df_GEO = df_best_dto[df_best_dto['PREPROC'] == '_Geometric_SMOTE'].copy()
    df_SMOTE = df_best_dto[df_best_dto['PREPROC'] == '_SMOTE'].copy()
    df_SMOTEsvm = df_best_dto[df_best_dto['PREPROC'] == '_smoteSVM'].copy()
    df_original = df_best_dto[df_best_dto['PREPROC'] == '_train'].copy()'''
    
    '''for o in order:
        for a in alphas:
            GEOMETRY = '_delaunay_'+ o + '_'+str(a)
            df_dto = df_best_dto[df_best_dto['PREPROC'] == GEOMETRY].copy()
            df = pd.concat([df_B1,df_B2,df_GEO,df_SMOTE,df_SMOTEsvm,df_original,df_dto])
            diag.rank_by_algorithm(df, 'multiclasse', './../output_dir/multiclass/', 'pca', o, str(a))
            diag.rank_dto_by(o + '_'+ str(a))'''
    ####################################
    '''o = 'solid_angle'
    a = 7.0
    GEOMETRY = '_delaunay_' + o + '_' + str(a)
    df_dto = df_best_dto[df_best_dto['PREPROC'] == GEOMETRY].copy()
    df = pd.concat([df_B1, df_B2, df_GEO, df_SMOTE, df_SMOTEsvm, df_original, df_dto])
    diag.rank_by_algorithm(df, 'multiclasse', './../output_dir/multiclass/', 'pca', o, str(a))
    diag.rank_dto_by(o + '_' + str(a))'''
    ###################################
    
    
    
    #diag.rank_total_by_algorithm('biclasse', './../output_dir/results_totais/', 'pca', 'max_solid_angle', 4)
    #diag.find_best_rank('./../output_dir/multiclass/media_rank/', 'best_pca_multiclass_media_rank.csv')
    #diag.find_best_delaunay('./../output_dir/', 'best_pca_multiclass_media_rank.csv')
    diag.grafico_variacao_alpha()
    



if __name__ == '__main__':
    main()