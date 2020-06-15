import numpy as np
import pandas as pd
import cv2
import random
import os


def remove_faulty(dataset, path):
    
	""" Removes the faulty images (path) from the dataset
	
	Args:
		dataset (dict): Full dataset
		path (str): Path of file containing the ID of faulty elements

	Returns:
		dataset (dict): Corrected dataset
	"""
	
	f = open(path, 'r')
	excludes = f.readlines()
	excludes_vec = np.ones(len(dataset['ids']), dtype="uint8") 
	excludes_pos = []

	for exclude in excludes:
		tmp = exclude.split("\n")[0]
		if "?" in tmp: # some marked with '?'
			tmp = tmp.split("?")[0]
		exclude_full = 'Train_' + tmp + '.jpg'
		for i, j in enumerate(dataset['ids']):
			if j == exclude_full:
				excludes_vec[i] = False
				excludes_pos = np.append(excludes_pos, i)
	excludes_pos = np.flip(np.sort(excludes_pos)).astype(int) # remove from the end always
	excludes_vec = excludes_vec == 1  # convert to bool

	for key in dataset.keys():
		if type(dataset[key]) is not str:
			dataset[key] = dataset[key][excludes_vec]
	
	return dataset


def summarize(dataset):

	""" Print information about the content of given dataset
	"""

	labels_int = np.argmax(dataset['y'], axis=1)
	labels_unique = np.unique(labels_int)
	labels_sum = np.zeros(len(labels_unique))
	print("All: " + str(dataset['x'].shape))
	# determine which class has the most sample and the target output
	for i in range(len(labels_unique)):
		label_cur = labels_unique[i]
		labels_sum[i] = sum(labels_int == label_cur)
		print("Label: " + str(label_cur) + " samples: " + str(labels_sum[i])) 


def normalize(dset, mean, std):

    """ Normalize the dataset. based on given parameters '

    Args: 
        train (ndarray): not normalized dataset
        mean (list): Mean for each channel (previously computed)
        std (list): std for each channel (previously computed)

    Returns:
        dset (dict): normalized dataset 
        mean (list): mean for each channel
        std (list): std for each channel

    """
    num_channels = 3
    if isinstance(mean, np.ndarray) and isinstance(std, np.ndarray):
        dset = np.cast['float32'](dset)
        for i in range(num_channels):
            dset[:,:,:,i] = (dset[:,:,:,i] - mean[i]) / std[i]
    else:	 
        # calculate mean and std over full training set
        mean = np.zeros(num_channels, dtype='float32')
        std = np.zeros(num_channels, dtype='float32')
        for i in range(num_channels):
            mean[i] = np.mean(dset[:,:,:,i])
            std[i] = np.std(dset[:,:,:,i])
        
    return dset, mean, std


def load(path, **kwargs):

	""" Loads the dataset from path

	Args: 
		path (str): Location of dataset

	Kwargs:	
		labels_path (str): Location of labels for the dataset (eq. training set)
		limit_flag (int): Load only [limit_flag] amount of image from the dataset

	Returns:
		dataset (dict): Dictionary containing every important information from dataset

	"""

	# defaults
	label_path = []
	limit_flag = False

	# kwargs
	if kwargs is not None:
		for key, value in kwargs.items():
			if key in 'labels_path':
				label_path = value
			if key in 'limit_flag':
				limit_flag = value
				
	# if dataset has labels
	labels_flag = False
	if isinstance(label_path, str): 
		labels_table = pd.read_csv(label_path)
		labels_flag = True

	### prepare for dataset loading				
	data_ids = os.listdir(path)
	limit = len(data_ids)
	num_classes = 4

	if limit_flag:
		limit = limit_flag
		data_ids = data_ids[0:limit]

	# example img to get imgsize
	for i in range(len(data_ids)):
		img0 = cv2.imread(os.path.join(path,data_ids[i]))
		if img0 is not None:
			imgsize = img0.shape
			break

	data_x = np.zeros((len(data_ids), imgsize[0], imgsize[1], imgsize[2]), dtype='float32')
	data_y = np.zeros((len(data_ids),num_classes), dtype="uint8")

	# load cycle
	for i in range(limit):
		img0 = cv2.imread(os.path.join(path,data_ids[i]))

		if img0 is not None:
			if i % 100 == 0:
				print(i)
			if labels_flag: # directly convert to one hot labels
				data_y[i] = (labels_table[['healthy', 'multiple_diseases', 'rust', 'scab']].values[data_ids[i].split('.')[0] == labels_table['image_id'].values])
			data_x[i] = img0
		else:
			print("Image Read Error")
			print(data_ids[i])

	# force data ids list to be an np.array
	data_ids = np.array(data_ids)

	# make a dictionary from arrays
	dataset = {	'x': data_x,
				'y': data_y,
				'ids': data_ids,
				'path': path}

	return dataset

	
def resample(dataset, which_class, how_many_times):

	""" Resample the desired class in case of skewed dataset
	"""

	# save variables
	vec = np.argmax(dataset['y'], axis = 1) == which_class
	ds2 = dataset.copy()

	# resampling
	for i in range(how_many_times):
		for key in ds2.keys():
			if type(ds2[key]) is not str:
				ds2[key] = np.concatenate((ds2[key], dataset[key][vec]), axis = 0)

	return ds2