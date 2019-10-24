import sys
import time

from folders import folder_experiments
from oversampling import Oversampling


def timer(start, end):
	hours, rem = divmod(end - start, 3600)
	minutes, seconds = divmod(rem, 60)
	print("{:0>2}:{:0>2}:{:0.1f}".format(int(hours), int(minutes), seconds))


def main():
	start = time.time()
	print('INIT')
	delaunaySMOTE = Oversampling()
	print('STEP 1')
	delaunaySMOTE.createValidationData(folder_experiments)
	print('STEP 2')
	#delaunaySMOTE.runSMOTEvariationsGen(folder_experiments)
	print('STEP 3')
	#delaunaySMOTE.runDelaunayVariationsGen(folder_experiments)
	print('STEP 4')
	#delaunaySMOTE.runClassification(folder_experiments,SMOTE=True)
		
	end = time.time()
	print("Total Execution Time : ")
	timer(start, end)


if __name__ == "__main__":
	main()
