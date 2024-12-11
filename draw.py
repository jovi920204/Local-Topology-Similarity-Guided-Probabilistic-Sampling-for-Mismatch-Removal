import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from FeatureExtraction import is_point_satisfying_homography

def plot_topology(img1Path, img2Path, source_points, target_points, source_tri, target_tri):
    img1 = cv.imread(img1Path, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(img2Path, cv.IMREAD_GRAYSCALE)
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    padded_img1 = np.pad(img1, ((0, max(0, h2 - h1)), (0, 0)), mode='constant')
    padded_img2 = np.pad(img2, ((0, max(0, h1 - h2)), (0, 0)), mode='constant')
    combined_image = np.hstack((padded_img1, padded_img2))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(combined_image, cmap='gray')
    ax.axis('off')
    for tri in source_tri.simplices:
        # plot the line in cv2 imaage 1
        pt1 = source_points[tri[0]]
        pt2 = source_points[tri[1]]
        pt3 = source_points[tri[2]]
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], '#00ff00', linewidth=0.5)
        ax.plot([pt2[0], pt3[0]], [pt2[1], pt3[1]], '#00ff00', linewidth=0.5)
        ax.plot([pt3[0], pt1[0]], [pt3[1], pt1[1]], '#00ff00', linewidth=0.5)
    for tri in target_tri.simplices:
        # plot the line in cv2 imaage 2
        pt1 = target_points[tri[0]]
        pt2 = target_points[tri[1]]
        pt3 = target_points[tri[2]]
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], '#00ff00', linewidth=0.5)
        ax.plot([pt2[0], pt3[0]], [pt2[1], pt3[1]], '#00ff00', linewidth=0.5)
        ax.plot([pt3[0], pt1[0]], [pt3[1], pt1[1]], '#00ff00', linewidth=0.5)
    plt.show()
    
def plot_match_points(img1Path, img2Path, source_points, target_points, is_corrected, H):
    img1 = cv.imread(img1Path, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(img2Path, cv.IMREAD_GRAYSCALE)
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    padded_img1 = np.pad(img1, ((0, max(0, h2 - h1)), (0, 0)), mode='constant')
    padded_img2 = np.pad(img2, ((0, max(0, h1 - h2)), (0, 0)), mode='constant')
    combined_image = np.hstack((padded_img1, padded_img2))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(combined_image, cmap='gray')
    ax.axis('off')
    for i in range(source_points.shape[0]):
        pt1 = source_points[i]
        pt2 = target_points[i]
        if is_corrected[i] == 1:
            if is_point_satisfying_homography(H, pt1[0], pt1[1], pt2[0], pt2[1]):
                color = '#00ff00'
            else:
                color = '#ff0000'
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color, linewidth=0.5)
    plt.show()
