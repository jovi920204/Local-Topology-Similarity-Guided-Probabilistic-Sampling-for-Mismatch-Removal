import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def is_point_satisfying_homography(H, x1, y1, x2, y2, tolerance=4):
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
    print(f"SIFT Number of matches: {len(good_matches)}")   


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
    print(f"True Number of inliers: {correct}")
    print(f"True Number of outliers: {mismatch}")
    return np.array(source_points), np.array(target_points), correct, mismatch
    
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