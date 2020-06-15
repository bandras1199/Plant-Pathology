import numpy as np
import random
import math
import cv2


def distort(img, orientation, x_scale, y_scale):

	""" Apply distortion on the original image.
		reference: https://medium.com/@schatty/image-augmentation-in-numpy-the-spell-is-simple-but-quite-unbreakable-e1af57bb50fd

		Args: 
			img (np.ndarray): Original Image
					orientation (str): Orientation of distortion, "horizontal" or "vertical"
					x_scale (int): Scaling in x direction
					y_scale (int): Scaling in y direction

		Returns:	
			img (np.ndarray): Distorted Image
	"""

	img_dist = img.copy()

	for c in range(3):
		for i in range(img.shape[orientation.startswith('ver')]):
			if orientation.startswith('ver'):
				img_dist[:, i, c] = np.roll(img[:, i, c], int(y_scale * np.sin(np.pi * i * x_scale)))
			else:
				img_dist[i, :, c] = np.roll(img[i, :, c], int(y_scale * np.sin(np.pi * i * x_scale)))
          
	return img_dist


def mirror(img):
	c = random.choice([0,1])
	return cv2.flip(img, c)


def rotate_autocrop(img, maxrotation):

	""" Rotate and automatically crop the image to remove black parts.
		reference: https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Operations.py

		Args: 
			img (np.ndarray): Original Image
					maxrotation (float): Maximum angle of allowed rotation 
			(it is selected randomly between 5 - max)

		Returns:	
			img (np.ndarray): Rotated and Cropped image
	"""

	rotation = random.choice(np.linspace(5,maxrotation))

	### rotation and extending image size
	x, y = img.shape[:2]
	image_center = (y/2, x/2)
	rotation_mat = cv2.getRotationMatrix2D(image_center, rotation, 1.)

	abs_cos = abs(rotation_mat[0,0]) 
	abs_sin = abs(rotation_mat[0,1])

	# new width and height bounds
	bound_w = int(x * abs_sin + y * abs_cos)
	bound_h = int(x * abs_cos + y * abs_sin)

	# new image center coordinates
	rotation_mat[0, 2] += bound_w/2 - image_center[0]
	rotation_mat[1, 2] += bound_h/2 - image_center[1]

	# rotate image with the new bounds and translated rotation matrix
	img = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h))

	# Get size after rotation, which includes the empty space
	X, Y = img.shape[:2]

	# Get our two angles needed for the calculation of the largest area
	angle_a = abs(rotation)
	angle_b = 90 - angle_a

	# Python deals in radians so get our radians
	angle_a_rad = math.radians(angle_a)
	angle_b_rad = math.radians(angle_b)

	# Calculate the sins
	angle_a_sin = math.sin(angle_a_rad)
	angle_b_sin = math.sin(angle_b_rad)

	# Find the maximum area of the rectangle that could be cropped
	E = (math.sin(angle_a_rad)) / (math.sin(angle_b_rad)) * \
		(Y - X * (math.sin(angle_a_rad) / math.sin(angle_b_rad)))
	E = E / 1 - (math.sin(angle_a_rad) ** 2 / math.sin(angle_b_rad) ** 2)
	B = X - E
	A = (math.sin(angle_a_rad) / math.sin(angle_b_rad)) * B

	# Crop this area from the rotated image
	y1 = int(round(E))
	x1 = int(round(A))
	y2 = int(round(X-E))
	x2 = int(round(Y-A))
	img = img[y1:y2, x1:x2]
	img = cv2.resize(img, (y,x), interpolation = cv2.INTER_CUBIC)

	return img


def pipeline(imageset,labels, **kwargs):

	""" Apply augmentation on the training dataset and balance each class.

		Args: 
			imageset (np.ndarray): Original Image Dataset
			labels (np.ndarray): Corresponding Labels (one-hot)

		Returns:	
			imageset_new (np.ndarray): Image Dataset after Augmentation
					labels_new (np.ndarray): Extended Labels (one-hot) after Augmentation

		Kwargs:		
			orientation (str): Orientation of distortion, "horizontal" or "vertical"
					x_scale (int): Scaling in x direction
					y_scale (int): Scaling in y direction
					maxrotation (float): Maximum angle of allowed rotation 
	"""

	imageset = imageset.copy()
	labels = labels.copy()
	orientation = 'ver'  # vertical or horizontal
	x_scale = 0.04
	y_scale = 4
	maxrotation = 15

	if kwargs is not None:
		for key, value in kwargs.items():
		#print(key, value)
		if key in 'ori':
			self.orientation = value
		elif key in 'x':
			self.x_scale = value
		elif key in 'y':
			self.y_scale = value
		elif key in 'rot':
			self.maxrotation = value

	labels_int = np.argmax(labels, axis=1)
	labels_unique = np.unique(labels_int)
	labels_sum = np.zeros(len(labels_unique), dtype='uint8')

	# determine which class has the most sample and the target output
	for i in range(len(labels_unique)):
		label_cur = labels_unique[i]
		labels_sum[i] = sum(labels_int == label_cur)

	labels_LargestClassSize = np.max(labels_sum)
	target = labels_LargestClassSize * 2
	target_dif = target - labels_sum

	# allocate in advance -> faster runtime
	imageset_new = np.zeros((imageset.shape[0]+sum(target_dif), imageset.shape[1], imageset.shape[2], imageset.shape[3]), dtype='uint8')
	labels_new = []
	c = 0

	for j in range(len(labels_unique)):
		label_cur = labels_unique[j]
		vec = labels_int == label_cur
		imagesubset = imageset[vec]

	for i in range(target_dif[j]):
		img = random.choice(imagesubset)
		img = distort(img, orientation, x_scale, y_scale)
		img = mirror(img)
		img = rotate_autocrop(img, maxrotation)

		imageset_new[c] = img
		labels_new.append(label_cur)
		c += 1

	imageset_new[c:] = imageset
	labels_new = np.append(labels_new, labels_int) # this way no new array 

	# convert labels back to one-hot
	n_labels = np.max(labels_int) + 1
	labels_new = np.eye(n_labels)[labels_new]

	return imageset_new, labels_new