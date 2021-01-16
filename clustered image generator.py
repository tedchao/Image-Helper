#!/usr/bin/env python3
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import cv2

def RGB_to_RGBXY( img, width, height ):
	'''
	Given: np array of images (shape: width x hegith, 3), image width, image height.   
	
	Return: An image with RGBXY. (shape: width x hegith, 5)
	'''
	
	img_rgbxy = np.zeros( ( img.shape[0], 5 ) )
	img = np.asfarray( img ).reshape( width, height, 3 )
	img_rgbxy = np.asfarray( img_rgbxy ).reshape( width, height, 5 )
	
	for i in range( width ):
		for j in range( height ):
			img_rgbxy[i, j, :3] = img[i, j]
			img_rgbxy[i, j, 3:5] = ( 1 / max( width, height ) ) * np.array( [i, j] )	# rescale spatial contribution
	img_rgbxy = img_rgbxy.reshape( ( -1, 5 ) )
	
	return img_rgbxy



def cv2_kmeans( input_path, output_path, num_clusters, option ):
	
	print( 'Start running Kmeans in CV2...' )
	image = cv2.imread( input_path )
		
	shape = image.shape
	
	image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )
	image = np.float32( image.reshape( ( -1, 3 ) ) )	# image data points
	
	criteria = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0 )
	
	attempts=10
	
	if option == 'RGBXY':
		image = np.float32( RGB_to_RGBXY( image, shape[0], shape[1] ) )
	
	ret, label, center = cv2.kmeans( image, num_clusters, None, criteria, attempts, cv2.KMEANS_PP_CENTERS )
	center = np.uint8( center )
	res = center[ label.flatten() ]
	
	if option == 'RGBXY':
		res = res[:, :3]
		
	result_image = res.reshape( ( shape ) )
	
	result_image = cv2.cvtColor( result_image, cv2.COLOR_RGB2BGR )
	
	cv2.imwrite( output_path + '-clustered' + option + str( num_clusters ) + '.jpg', result_image )
	print( 'Done Kmeans CV2!' )
	
	
def main():
	import argparse
	parser = argparse.ArgumentParser( description = 'Analysis of Kmeans.' )
	parser.add_argument( 'input_image', help = 'The path to the input image.' )
	parser.add_argument( 'output_path', help = 'Where to save the output clustered image.' )
	parser.add_argument( 'numK', help = 'Number of clusters.' )
	parser.add_argument( 'option', help = 'Kmeans in RGB or RGBXY' )
	args = parser.parse_args()
	
	
	img_arr = np.asfarray( Image.open(args.input_image).convert( 'RGB' ) ) / 255.
	# algorithm starts
	start = time.time()

	# kmeans for cv2
	cv2_kmeans( args.input_image, args.output_path, int( args.numK ), args.option )
		
	end = time.time()
	print( "Finished. Total time: ", end - start )
	print( '----------------------------' )
	
	
if __name__ == '__main__':
	main()
	
