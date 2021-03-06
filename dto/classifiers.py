from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

base_estimator = AdaBoostClassifier(n_estimators=10)

classifiers = {"RF": RandomForestClassifier(n_estimators=100),
			   "KNN": KNeighborsClassifier(),
			   "DTREE": DecisionTreeClassifier(),
			   "LRG": LogisticRegression(),
			   "ABC": AdaBoostClassifier(),
			   "MLP": MLPClassifier(max_iter=500),
			   "SVM": SVC(probability=True),
			   "SGD": SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
			   }

classifiers_list = ['RF','KNN','DTREE','LRG','ABC','MLP','SVM','SGD']
