import numpy as np
import dlib
import cv2
import imageio

import utilities as U
import sys
import getopt


if __name__ == '__main__':

	####### handle inputs ########
	try:
		opts, args = getopt.getopt(sys.argv[1:], "hs:t:f:", ["source=","target=","format="])
	except getopt.GetoptError as err:
		print(err)
		sys.exit('Error Occurs ')


	img_source = None
	img_target = None
	out_format = None
	for opt, arg in opts:
		if opt == '-h':
			print('Usage:')
			print('Short: python3 morphing.py -s <image1> -t <image2> -f <gif> or mp4>')
			print('Long: python3 morphing.py --source <image1> --target <image2> --format <gif> or mp4>')
			sys.exit()

		if opt in ("-s", "--source"):
			if arg.split('.')[-1] in ["jpeg","jpg","png"]:
				img_source = arg

		if opt in ("-t", "--target"):
			if arg.split('.')[-1] in ["jpeg","jpg","png"]:
				img_target = arg

		if opt in ("-f", "--format"):
			if arg in ["gif","mp4"]:
				out_format = arg


	if not img_source: sys.exit('Wrong source image format: only support jpeg, jpg, png')
	if not img_target: sys.exit('Wrong source image format: only support jpeg, jpg, png')
	if not out_format: sys.exit('Wrong output format: only support gif, mp4')


	###### start preprocess ########
	print('Step 1: Start to find corresponding points...')

	# create detector and predictor for finding the facial feature
	detector = dlib.get_frontal_face_detector()
	# pre-trained model: shape_predictor_68_face_landmarks.dat, this file is stored in the same directory
	predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

	# source image
	img1 = cv2.imread(img_source)
	img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	# target image
	img2 = cv2.imread(img_target)
	img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	source_points = U.find_68_facial_landmarks(img1_gray, detector, predictor)
	target_points = U.find_68_facial_landmarks(img2_gray, detector,predictor)


	# Read array of corresponding points
	points1 = source_points.tolist()
	points2 = target_points.tolist()
	# Append 8 additional points: corners and half way points
	size = img1.shape
	h,w = size[0], size[1]
	h2, w2 = int(size[0]/2), int(size[1]/2)
	points1.append( [0    , 0    ] )
	points1.append( [0    , h - 1] )
	points1.append( [w - 1, 0    ] )
	points1.append( [w - 1, h - 1] )
	points1.append( [0    , h2   ] )
	points1.append( [w2   , 0    ] )
	points1.append( [w - 1, h2   ] )
	points1.append( [w2   , h - 1] )

	size = img2.shape
	h,w = size[0], size[1]
	h2, w2 = int(size[0]/2), int(size[1]/2)
	points2.append( [0    , 0    ] )
	points2.append( [0    , h - 1] )
	points2.append( [w - 1, 0    ] )
	points2.append( [w - 1, h - 1] )
	points2.append( [0    , h2   ] )
	points2.append( [w2   , 0    ] )
	points2.append( [w - 1, h2   ] )
	points2.append( [w2   , h - 1] )


	print('Step 1 Finished.')


	###### Compute Delaunay triangluation ########
	print('Step 2: Find Delaunay triangles for corresponding points')
	delaunay = U.get_delaunay_indexes(img1,points1)
	print('Step 2 Finished.')


	###### Morphing the image ########
	print('Step 3: Start morphing!')
	# Alpha values
	alpha_values = np.linspace(0, 100, 30)  # 30 fps
	
	image_out = []
	for (f, a) in enumerate(alpha_values):

	    alpha = float(a) / 100
	    
	    points = []
	    # Compute weighted average point coordinates for morphed image
	    for i in range(0, len(points1)):
	        x = ( 1 - alpha ) * points1[i][0] + alpha * points2[i][0]
	        y = ( 1 - alpha ) * points1[i][1] + alpha * points2[i][1]
	        points.append((x,y))

	    # initialize output image
	    imgMorph = np.zeros(img1.shape, dtype = img1.dtype)

	    for v1, v2, v3 in delaunay :
	        t1 = [points1[v1], points1[v2], points1[v3]]
	        t2 = [points2[v1], points2[v2], points2[v3]]
	        t  = [ points[v1],  points[v2],  points[v3]]

	        # Morph one triangle at a time.
	        U.morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)


	    # Save morphing frame, uncomment this if want to see the all the frames
	    # index = str(f).zfill(4)
	    # cv2.imwrite(f'img-{index}.png', np.uint8(imgMorph))
	    
	    # Save frame into a sequence
	    # convert opencv image color BGR to skimage color RGB
	    out_im = cv2.cvtColor(np.uint8(imgMorph), cv2.COLOR_BGR2RGB)
	    image_out.append(out_im)
	    
	    
	# generate gif and mp4
	if out_format == 'gif':
		imageio.mimwrite('morphed_image.gif', image_out)
	if out_format == 'mp4':
		imageio.mimwrite('morphed_image.mp4', image_out)


	print('Step 3: Finished. Done!')








