#!/usr/bin/env python3
import numpy as np
from PIL import Image
import scipy.misc as scp

def preprocessing(path):

	image = Image.open(path).convert('RGB') # load an image
	image = np.asarray(image) # convert to a numpy array

	row, col, depth= image.shape

	if row>col:
		image_square = np.zeros([row,row,3])
	else:
		image_square = np.zeros([col,col,3])
	# padding
	for idx in range(depth):
		if row>col:
			padding = ((0,0),((row-col)//2,(row-col)//2))
		else:
			padding = (((col-row)//2,(col-row)//2),(0,0))

		paddedImg = np.pad(image[:,:,idx],padding,'constant', constant_values = 0)
		image_square[0:paddedImg.shape[0],0:paddedImg.shape[1],idx] = paddedImg
	#rescaling
	image_square_resized = scp.imresize(image_square,(224,224))

	#normalization
	image_square_resized = image_square_resized / 255

	return image_square_resized
