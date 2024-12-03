import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay 
import os

def SIFT(img1Path, img2Path, H, threshold=0.8, isShow=False):
    sift = cv.SIFT_create()
    img1 = cv.imread(img1Path, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(img2Path, cv.IMREAD_GRAYSCALE)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)
    print(f"Number of matches: {len(good_matches)}")   


    # 創建顯示用的拼接圖像
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    padded_img1 = np.pad(img1, ((0, max(0, h2 - h1)), (0, 0)), mode='constant')
    padded_img2 = np.pad(img2, ((0, max(0, h1 - h2)), (0, 0)), mode='constant')
    combined_image = np.hstack((padded_img1, padded_img2))

    source_points = []
    target_points = []
    correct = 0
    mismatch = 0
    if isShow:
        # 繪製匹配結果
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(combined_image, cmap='gray')
        ax.axis('off')

        # 繪製匹配點和連線
        for match in good_matches:
            # 獲取特徵點座標
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            pt2 = (pt2[0] + w1, pt2[1])  # 偏移第二張圖片的 x 座標
            source_points.append([pt1[0], pt1[1]])
            target_points.append([pt2[0], pt2[1]])
            
            ### pt1 * H == pt2
            check = is_point_satisfying_homography(H, pt1[0], pt1[1], pt2[0], pt2[1])
            if (check):
                color = '#00ff00'
                correct += 1
            else:
                color = '#ff0000'
                mismatch += 1
                
            # 繪製連線
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color, linewidth=0.5)
        plt.show()
    else:
        for match in good_matches:
            # 獲取特徵點座標
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            pt2 = (pt2[0] + w1, pt2[1])  # 偏移第二張圖片的 x 座標
            source_points.append([pt1[0], pt1[1]])
            target_points.append([pt2[0], pt2[1]])
            ### pt1 * H == pt2
            check = is_point_satisfying_homography(H, pt1[0], pt1[1], pt2[0], pt2[1])
            if (check):
                correct += 1
            else:
                mismatch += 1   
    print(f"Number of inliers: {correct}")
    print(f"Number of outliers: {mismatch}")
    return np.array(source_points), np.array(target_points)
    
def ORB(img1Path, img2Path, H, threshold=0.9572, isShow=False):
    orb = cv.ORB_create()
    img1 = cv.imread(img1Path, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(img2Path, cv.IMREAD_GRAYSCALE)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)
    print(f"Number of matches: {len(good_matches)}")   


    # 創建顯示用的拼接圖像
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    padded_img1 = np.pad(img1, ((0, max(0, h2 - h1)), (0, 0)), mode='constant')
    padded_img2 = np.pad(img2, ((0, max(0, h1 - h2)), (0, 0)), mode='constant')
    combined_image = np.hstack((padded_img1, padded_img2))

    source_points = []
    target_points = []
    correct = 0
    mismatch = 0
    if isShow:
        # 繪製匹配結果
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(combined_image, cmap='gray')
        ax.axis('off')
        # 繪製匹配點和連線
        for match in good_matches:
            # 獲取特徵點座標
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            pt2 = (pt2[0] + w1, pt2[1])  # 偏移第二張圖片的 x 座標
            source_points.append([pt1[0], pt1[1]])
            target_points.append([pt2[0], pt2[1]])
            ### pt1 * H == pt2
            check = is_point_satisfying_homography(H, pt1[0], pt1[1], pt2[0], pt2[1])
            if (check):
                color = '#00ff00'
                correct += 1
            else:
                color = '#ff0000'
                mismatch += 1

            # 繪製連線
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color, linewidth=0.5)
        plt.show()
    else:
        for match in good_matches:
            # 獲取特徵點座標
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            pt2 = (pt2[0] + w1, pt2[1])  # 偏移第二張圖片的 x 座標
            source_points.append([pt1[0], pt1[1]])
            target_points.append([pt2[0], pt2[1]])
            ### pt1 * H == pt2
            # check = is_point_satisfying_homography(H, pt1[0], pt1[1], pt2[0], pt2[1])
            # if (check):
            #     correct += 1
            # else:
            #     mismatch += 1
    print(f"Number of inliers: {correct}")
    print(f"Number of outliers: {mismatch}")
    return np.array(source_points), np.array(target_points)

def is_point_satisfying_homography(H, x1, y1, x2, y2, tolerance=0.41):
    """
    檢查 (x1, y1) 和 (x2, y2) 是否滿足單應矩陣 H。

    參數:
        H (numpy.ndarray): 3x3 單應矩陣。
        x1, y1 (float): 第一張圖片的點座標。
        x2, y2 (float): 第二張圖片的點座標。
        tolerance (float): 判斷誤差容許範圍。

    返回:
        bool: 是否滿足單應矩陣 H 的映射關係。
    """
    # 將 (x1, y1) 表示為齊次座標
    point1 = np.array([x1, y1, 1.0])
    
    # 使用單應矩陣進行映射
    mapped_point = np.dot(H, point1)
    if mapped_point[2] == 0 or mapped_point[2] == 0:
        return False
    # 將結果轉換為非齊次座標
    x2_mapped = mapped_point[0] / mapped_point[2]
    y2_mapped = mapped_point[1] / mapped_point[2]
    # print(abs(x2 - x2_mapped), abs(y2 - y2_mapped))
    # 檢查 (x2, y2) 是否接近映射結果
    if abs(x2 - x2_mapped) <= tolerance or abs(y2 - y2_mapped) <= tolerance:
        return True
    else:
        return False
    
def construct_adj(connect_info, size):
    """
    Construct the adjacent matrix of the image
    """
    adj = np.zeros((size, size))
    for tri in connect_info:
        for i in range(3):
            for j in range(i+1, 3):
                adj[tri[i], tri[j]] = 1
                adj[tri[j], tri[i]] = 1
    return adj

def calculate_probability(SUM, average, sigma):
    """
    Calculate the probability of the SUM
    """
    p = np.zeros(SUM.shape)
    for i in range(SUM.shape[0]):
        p[i] = 1 - 1/(1 + np.exp((-SUM[i] - average) / sigma))
    return p
    
def sample_probability(p):
    """
    Sample the probability
    """
    total_probability = 0
    for i in range(p.shape[0]):
        total_probability += (1 - p[i])
    ps = np.zeros(p.shape)
    for i in range(p.shape[0]):
        ps[i] = (1 - p[i]) / total_probability        
    return ps

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

if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    img1Path = os.path.join(root, "dataset/boat/img1.pgm")
    img2Path = os.path.join(root, "dataset/boat/img3.pgm")
    groundTruthPath = os.path.join(root, "dataset/boat/H1to3p")
    
    ### open the ground truth file stored in Homography format
    with open(groundTruthPath, 'r') as f:
        lines = f.readlines()
        H = np.array([list(map(float, line.split())) for line in lines])
    # print(H)
    source_points, target_points = SIFT(img1Path, img2Path, H, threshold=0.2834)
    # source_points, target_points = ORB(img1Path, img2Path, H, threshold=1, isShow=False)
    number_of_points = source_points.shape[0]
    
    # construct the topology of the two images
    source_tri = Delaunay(source_points)
    target_tri = Delaunay(target_points)
    
    # construct the adjecent matrix of the two images
    source_m = construct_adj(source_tri.simplices, number_of_points)
    target_m = construct_adj(target_tri.simplices, number_of_points)
    
    # element-wise multiplication
    m = np.multiply(source_m, target_m)

    # SUM
    one_vector = np.ones(number_of_points)
    SUM = np.dot(one_vector, m)
    
    # probability
    average, sigma = np.mean(SUM), np.std(SUM)
    p = calculate_probability(SUM, average, sigma)

    # sampling probability
    ps = sample_probability(p)
    
    # RANSAC according to the ps(Probability sampling)
    threshold = 0.16
    max_iteration = 1000
    max_inliers = 0
    max_H = None
    max_is_corrected = None
    for i in range(max_iteration):
        is_corrected = np.zeros(number_of_points)
        # random sample
        sample = np.random.choice(number_of_points, 4, replace=False, p=ps)
        # calculate the homography matrix
        curr_H = cv.findHomography(source_points[sample], target_points[sample])[0]
        # calculate the inliers
        inliers = 0
        for i in range(number_of_points):
            check = is_point_satisfying_homography(curr_H, source_points[i][0], source_points[i][1], target_points[i][0], target_points[i][1], tolerance=threshold)
            if check:
                is_corrected[i] = 1
                inliers += 1
        if inliers > max_inliers:
            max_inliers = inliers
            max_H = curr_H
            max_is_corrected = is_corrected
    print(f"Number of max inliers: {max_inliers}")
    
    plot_match_points(img1Path, img2Path, source_points, target_points, max_is_corrected, H)
    # plot_topology(img1Path, img2Path, source_points, target_points, source_tri, target_tri)