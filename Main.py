import sys
import time

from folders import folder_experiments
from oversampling import Oversampling

sys.path.append('/home/amc/Doutorado2019/V11_PCA_BICLASS_BEST_DELAUNAY/')

def timer(start, end):
	hours, rem = divmod(end - start, 3600)
	minutes, seconds = divmod(rem, 60)
	print("{:0>2}:{:0>2}:{:0.1f}".format(int(hours), int(minutes), seconds))


def main():
	start = time.time()
	print('Iniciar')
	delaunaySMOTE = Oversampling()
	print('Passo 1')
	#delaunaySMOTE.createValidationData(folder_experiments)
	print('Passo 2')
	#delaunaySMOTE.runSMOTEvariationsGen(folder_experiments)
	
	delaunaySMOTE.cria_graficos()
	
	#print('Passo 3')
	#delaunaySMOTE.runDelaunayVariationsGen(folder_experiments)
	#print('Passo 4')
	#delaunaySMOTE.runClassification(folder_experiments,SMOTE=True)
	
	
	
	
	
	end = time.time()
	print("Total Execution Time : ")
	timer(start, end)


if __name__ == "__main__":
	main()
