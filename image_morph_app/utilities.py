
import numpy as np
import cv2

# convert dlib format bounding box to opencv format bounding box
def rect_to_bb(rect):
    # to the format (x, y, width, height) as we would normally do
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    
    return (x, y, w, h)

# convert detected facial points to np.array format
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)  # only 68 points
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# detect 68 points of face
def find_68_facial_landmarks(image, detector, predictor):
    """ 1. Image is a gray scale with value 0-255. Otherwise, the detector won't work
        2. detector and predictor are pre-defined and loaded by dlib"""
    # detect the bounding box of face
    rects = detector(image, 1)
    # rects contains all the bounding boxes for all faces. if only one face, the loop only iterates once.
    for (i, rect) in enumerate(rects):
        # find the facial landmarks, then convert it to a numpy array
        points = predictor(image, rect)
        points = shape_to_np(points)  
            
    return points


# Gets delaunay 2D segmentation and return a list with the the triangles' indexes
def get_delaunay_indexes(image, points) :

    rect = (0, 0, image.shape[1], image.shape[0])
    subdiv = cv2.Subdiv2D(rect);
    for p in points :
        subdiv.insert( tuple(p) )

    triangleList = subdiv.getTriangleList();
    triangles = []
    for p in triangleList:
        vertexes = [0, 0, 0]
        for v in range(3) :
            cur_v = v * 2
            for i in range(len(points)) :
                if p[cur_v] == points[i][0] and p[cur_v+1] == points[i][1]:
                    vertexes[v] = i

        triangles.append(vertexes)

    return triangles

# Apply affine transform calculated using srcTri and dstTri to src and
def applyAffineTransform(src, srcTri, dstTri, size) :
    # Given a pair of triangles, find the affine transform.
    M = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, M, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))


    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))


    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask
    