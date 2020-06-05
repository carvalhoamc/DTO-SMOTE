import sys
import time

from datasetsDelaunay import dataset_multiclass
from oversampling import Oversampling


def timer(start, end):
	hours, rem = divmod(end - start, 3600)
	minutes, seconds = divmod(rem, 60)
	print("{:0>2}:{:0>2}:{:0.1f}".format(int(hours), int(minutes), seconds))

#Parameters
dataset_folder = './../datasets/'


def main():
	start = time.time()
	print('INIT')
	DATASET = dataset_multiclass #dataset_biclass or dataset_multiclass
	experiment_oversampling = Oversampling()
	print('Create Validation Data')
	experiment_oversampling.createValidationData(dataset_folder,datasets=DATASET)
	print('Run SMOTE Variations')
	experiment_oversampling.runSMOTEvariationsGen(dataset_folder,datasets=DATASET)
	print('Run DTOSMOTE')
	experiment_oversampling.runDelaunayVariationsGen(dataset_folder,datasets=DATASET)
	print('Run Classifiers')
	experiment_oversampling.runClassification(dataset_folder,datasets=DATASET,kind='multiclass')
	
	end = time.time()
	print("Total Execution Time : ")
	timer(start, end)


if __name__ == "__main__":
	main()
