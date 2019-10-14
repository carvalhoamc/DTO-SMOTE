from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsembleClassifier, RUSBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from dialnd_imbalanced_algorithms.smote import SMOTEBoost

base_estimator = AdaBoostClassifier(n_estimators=10)

classifiers = {"RF": RandomForestClassifier(n_estimators=100),#38 ocorrencias pca biclasse
			   "KNN": KNeighborsClassifier(),#20 ocorrencias pca biclasse
			   "DTREE": DecisionTreeClassifier(),#apenas 3 ocorrencias pca biclasse
			   "GNB": GaussianNB(),#apenas 1 ocorrencias pca biclasse
			   "LRG": LogisticRegression(),#nenhuma ocorrencias pca biclasse
			   "ABC": AdaBoostClassifier(),#22 ocorrencias pca biclasse
			   "MLP": MLPClassifier(max_iter=500),#24 ocorrencias pca biclasse
			   "QDA": QuadraticDiscriminantAnalysis(store_covariance=True),#46 ocorrencias pca biclasse
			   "SVM": SVC(probability=True),#apenas 1 ocorrencias pca biclasse
			   "SGD": SGDClassifier(loss="hinge", penalty="l2", max_iter=5)#10 ocorrencias pca biclasse
			   }
