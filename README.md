# Image morphing


Image morph is a special effect in motion pictures and animations that changes one image into another through a seamless transition. It is a very interesting application of computer vision that interests me to dive into deeply about how it works. In this project, I focus on how Image morph works and how to implement.

Notebooks folder contains experiments on finding facial landmarks, Delaunay Triangulation, and Voronoi Diagram.

# Data usage:

1. Two images: Source image is the starting image. Target image is the ending image.

2. Dlib python library for facial detection and making 68 facial landmarks.

3. A pre-train model for making landmarks: shape_predictor_68_face_landmarks.dat

4. Additional library: numpy, cv2, imageio.


# Procedure

1. Find corresponding points of facial landmark for both images.

2. Find the Delaunay Triangulation for those corresponding points.

3. Compute Affine transformation between the Delaunay triangles of both faces

4. Perform warping and alpha blending for initial image and final image.

5. Create a video/gif from newly created frames to show the morphing effect.


# Alpha Results

![alt text](https://github.com/zzyrd/Image_morphing/blob/master/alpha_result.png "result")

# How to use

1. Open the app folder:
	cd image_morph_app

2. Now see two images: christ-pratt.jpg, kobe-bryant.jpg
   One image is source image, Another one is target image. You can replace those by your own images

   Two python scripts:
   	morphying.py is our executable script
   	utilities.py stores all self-defined functions

3. Run morphing.py:

Help Command

    python3 morphing.py -h
	Usage:
	Short: python3 morphing.py -s <image1> -t <image2> -f <gif> or mp4>
	Long: python3 morphing.py --source <image1> --target <image2> --format <gif> or mp4>


Generate mp4 output


    python3 morphing.py -s christ-pratt.jpg -t kobe-bryant.jpg -f mp4

Generate gif output

	python3 morphing.py -s christ-pratt.jpg -t kobe-bryant.jpg -f gif

# Results
![alt text](https://github.com/zzyrd/Image_morphing/blob/master/morphed_image.gif "result")

# Reference & Credits:
	https://www.learnopencv.com/facial-landmark-detection/
	https://www.learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/
	https://www.learnopencv.com/face-morph-using-opencv-cpp-python/
