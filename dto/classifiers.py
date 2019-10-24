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

classifiers = {"RF": RandomForestClassifier(n_estimators=100),
			   "KNN": KNeighborsClassifier(),
			   "DTREE": DecisionTreeClassifier(),
			   "GNB": GaussianNB(),
			   "LRG": LogisticRegression(),
			   "ABC": AdaBoostClassifier(),
			   "MLP": MLPClassifier(max_iter=500),
			   "QDA": QuadraticDiscriminantAnalysis(store_covariance=True),
			   "SVM": SVC(probability=True),
			   "SGD": SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
			   }
