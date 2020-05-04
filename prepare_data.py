import cv2
import numpy as np
import matplotlib.pyplot as plt
from sympy import *


def FitEllipse(img):

	"""
		- Removes blurry background using Fourier Transformation
		- Detect contours around the focus
		- Fits an ellipse around the leaf (theoretically..)
		
		parameters: img (np.ndarray): Original Image
		returns:	masked_image (np.ndarray): Image masked with ellipse
					ellipse (tuple): parameters for cv2.ellipse()
	"""

	### remove blurry background 
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ft = np.fft.fft2(img_gray)
	fshift = np.fft.fftshift(ft)
	#magnitude_spectrum = 20*np.log(np.abs(fshift))
	rows, cols = img_gray.shape
	crow,ccol = int(rows/2) , int(cols/2)
	cover = 70 # size of the cover placed in the middle of the of magnitude spectrum
	fshift[crow-cover:crow+cover, ccol-cover:ccol+cover] = 0
	f_ishift = np.fft.ifftshift(fshift)
	img_back = np.fft.ifft2(f_ishift)
	img_back = np.uint8(np.abs(img_back))

	### crop because findcontours is prone to find contours on the side
	rows10 = int(rows*0.1)
	cols10 = int(cols*0.1)
	crow,ccol = int(rows/2) , int(cols/2)
	img_back = img_back[rows10:rows-rows10, cols10:cols-cols10]
	img = img[rows10:rows-rows10, cols10:cols-cols10]

	### highlighting contours
	thr = cv2.adaptiveThreshold(img_back,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
		cv2.THRESH_BINARY,11,2)
	thr = cv2.GaussianBlur(thr, (5,5), 0)
	thr_t = cv2.bitwise_not(thr)

	kernel = np.ones((5,5),np.uint8)
	erosion = cv2.erode(thr_t, kernel,iterations = 1)
	dilation = cv2.dilate(erosion,kernel,iterations = 1)
	final = dilation

	### contour detection
	cnt, _ = cv2.findContours(final,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cnt_sorted = sorted(cnt, key=lambda x: cv2.contourArea(x), reverse=True)

	ellipse = cv2.fitEllipse(cnt_sorted[0])  # direct?
	ellipse_mask = cv2.ellipse(np.zeros_like(img), ellipse, (255,255,255),-1)
	masked_image = cv2.bitwise_and(img, ellipse_mask)

	M = cv2.getRotationMatrix2D(ellipse[0], ellipse[2]-90, 1) # ellipse takes Y axes as 0 degree (-90), rotation center is ellipse!
	masked_image = cv2.warpAffine(masked_image.copy(), M, (img.shape[1], img.shape[0])) 

	## testing purposes
	#test_sorted = cv2.drawContours(img.copy(), cnt_sorted[0], -1, (255, 0, 0), 3)
	#plt.imshow(test_sorted, 'brg')
	#plt.show()

	return masked_image, ellipse


def FitRectangleInsideEllipse(img, ellipse, size):

	"""
		UPDATE:
		- use fixed aspect ratio for later resize of the image
		
		- fits the biggest possible rectangle inside the ellipse
		- f derivative reference: https://www.youtube.com/watch?v=r0wdreyN4QE
		
		parameters: img (np.ndarray): Original Image
					ellipse (tuple): parameters for cv2.ellipse()
					size (tuple): desired size of the output image
		returns:	masked_image (np.ndarray): Image masked with rectangle
					(r_start_point, r_end_point) (tuple): parameters for cv2.rectangle()
	"""

	img_size_desired_ratio = size[0] / size[1]

	e_centerCoordinates = ellipse[0]
	e_axesLength = ellipse[1]
	e_angle = ellipse[2] # not used?
	e_x_hlength = (e_axesLength[np.argmax(e_axesLength)] / 2) # a
	e_y_hlength = (e_axesLength[np.argmin(e_axesLength)] / 2) # b

	x = Symbol('x')
	f = (4*e_y_hlength*x)*sqrt(1-((x**2)/(e_x_hlength**2)))
	f_prime = f.diff(x)
	f = lambdify(x, f)
	f_prime = lambdify(x, f_prime)

	r_x_range = np.logspace(np.log10(min(e_x_hlength,e_y_hlength)/2), np.log10(min(e_x_hlength,e_y_hlength)), num=10)
	for r_x_hlength in r_x_range:
		if f_prime(r_x_hlength) < 0.5:
			break

	#r_y_hlength = f(r_x_hlength) / (4*r_x_hlength) # original computation
	# UPDATE: changed to keep fixed aspect ratio
	r_y_hlength = int(r_x_hlength / img_size_desired_ratio) 
	r_start_point = FloatToInt((e_centerCoordinates[0] - r_x_hlength, e_centerCoordinates[1] + r_y_hlength))
	r_end_point = FloatToInt((e_centerCoordinates[0] + r_x_hlength, e_centerCoordinates[1] - r_y_hlength))

	# UPDATE: fixed aspect ratio causes negative pixels, shift image:
	r_start_point = list(r_start_point)
	r_end_point = list(r_end_point)
	if r_start_point[0] < 0: # shift right
		r_end_point[0] = int(r_end_point[0] + abs(r_start_point[0]))
		r_start_point[0] = 0
	if r_end_point[1] < 0: # shift up
		r_start_point[1] = int(r_start_point[1] + abs(r_end_point[1]))
		r_end_point[1] = 0
	r_start_point = tuple(r_start_point)
	r_end_point = tuple(r_end_point)

	rectangle_mask = cv2.rectangle(np.zeros_like(img), r_start_point, r_end_point, (255,255,255), -1)
	masked_image = cv2.bitwise_and(img, rectangle_mask)

	return masked_image, (r_start_point, r_end_point)


def CropAroundRectangle(img, rectangle, size):

	"""
		- crop the image around the rectangle and resize to standard size
		
		parameters: img (np.ndarray): Original Image
					rectangle (tuple): parameters for cv2.rectangle()
					size (tuple): desired size of the output image
		returns:	img2 (np.ndarray): standard size image
					
	"""
    
	r_w = rectangle[1][0] - rectangle[0][0]
	img2 = img[rectangle[1][1]:rectangle[0][1], rectangle[0][0]:rectangle[1][0]]

	if r_w < size[0]: # enlarge
		img2 = cv2.resize(img2, size, interpolation = cv2.INTER_CUBIC)
	elif r_w > size[0]: # shrink
		img2 = cv2.resize(img2, size, interpolation = cv2.INTER_AREA)

	return img2


def SpectralFilter(img):

	"""
		UPDATE: not used
		- removes unnecessary greens from the image
		
		parameters: img (np.ndarray): Original Image
		returns:	res (np.ndarray): Spectral filtered image
					
	"""
  
	# maybe apply blur at the beggining?
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	lower_green = np.array([10,20,20])
	upper_green = np.array([100,200,200])

	mask = cv2.inRange(hsv, lower_green, upper_green)
	mask = cv2.bitwise_not(mask) # negate mask to remove green
	res = cv2.bitwise_and(img, img, mask = mask)

	return res

def NotNegative(num):
	if num < 0:
		num = 0
	return num

def FloatToInt(nums):

	"""
		Convert float type tuple coordinates to int type coordinates
	"""
	if len(nums) == 2:
		return (int(nums[0]),int(nums[1]))
	elif len(nums) == 3:
		return (int(nums[0]),int(nums[1]), int(nums[2]))
	else:
		return (int(nums))