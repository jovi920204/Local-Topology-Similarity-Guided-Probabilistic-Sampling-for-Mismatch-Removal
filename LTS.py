import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import os
from FeatureExtraction import is_point_satisfying_homography, SIFT, ORB
from draw import plot_topology, plot_match_points

class LTS:
    def __init__(self, img1_path, img2_path, ground_truth_path):
        self.img1_path = img1_path
        self.img2_path = img2_path
        self.ground_truth_path = ground_truth_path
        self.source_points = None
        self.target_points = None
        self.total_correct_matches = None # 在 ground truth 中的 correct match 數量
        self.total_mismatches = None # 在 ground truth 中的 mismatch 數量
        self.number_of_points = None # 總共的 feature points 數量
        self.extracted_matches = 0 # 經過 RANSAC 後的 match 數量, 不論正確與否
        self.correct_matches = 0 # 經過 RANSAC 後的 correct match 數量
        self.H = None
        self._load_ground_truth()

    def _load_ground_truth(self):
        """Load the ground truth homography matrix."""
        with open(self.ground_truth_path, 'r') as f:
            lines = f.readlines()
            self.H = np.array([list(map(float, line.split())) for line in lines])

    @staticmethod
    def construct_adj(connect_info, size):
        """Construct the adjacency matrix of the image."""
        adj = np.zeros((size, size))
        for tri in connect_info:
            for i in range(3):
                for j in range(i + 1, 3):
                    adj[tri[i], tri[j]] = 1
                    adj[tri[j], tri[i]] = 1
        return adj

    @staticmethod
    def calculate_probability(SUM, average, sigma):
        """Calculate the probability of the SUM."""
        return 1 - 1 / (1 + np.exp((-SUM - average) / sigma))

    @staticmethod
    def sample_probability(p):
        """Sample the probability."""
        total_probability = np.sum(1 - p)
        return (1 - p) / total_probability

    def extract_features(self, method="SIFT", threshold=1):
        """Extract features using SIFT or ORB.
        params:
            threshold: the threshold for matching points.
        """
        if method == "SIFT":
            self.source_points, self.target_points, self.total_correct_matches, self.total_mismatches = SIFT(
                self.img1_path, self.img2_path, self.H, threshold
            )
        elif method == "ORB":
            self.source_points, self.target_points, self.total_correct_matches, self.total_mismatches = ORB(
                self.img1_path, self.img2_path, self.H, threshold
            )
        else:
            raise ValueError("Unsupported feature extraction method.")
        self.number_of_points = self.source_points.shape[0]

    def run_ransac(self, max_iteration=100):
        """Run RANSAC to find the best homography matrix."""
        
        # Construct the adjacency matrix
        source_tri = Delaunay(self.source_points)
        target_tri = Delaunay(self.target_points)

        source_m = self.construct_adj(source_tri.simplices, self.number_of_points)
        target_m = self.construct_adj(target_tri.simplices, self.number_of_points)

        m = np.multiply(source_m, target_m)
        one_vector = np.ones(self.number_of_points)
        SUM = np.dot(one_vector, m)

        # Calculate the mismatching probability
        average, sigma = np.mean(SUM), np.std(SUM)
        p = self.calculate_probability(SUM, average, sigma)
        ps = self.sample_probability(p)

        max_inliers = 0
        max_H = None
        max_is_corrected = None

        # Probability guided sampling
        for _ in range(max_iteration):
            is_corrected = np.zeros(self.number_of_points)
            sample = np.random.choice(self.number_of_points, 4, replace=False, p=ps)
            curr_H = cv.findHomography(self.source_points[sample], self.target_points[sample])[0]

            inliers = 0
            for i in range(self.number_of_points):
                check = is_point_satisfying_homography(
                    curr_H,
                    self.source_points[i][0],
                    self.source_points[i][1],
                    self.target_points[i][0],
                    self.target_points[i][1],
                )
                if check:
                    is_corrected[i] = 1
                    inliers += 1
                    
            # 若 inliers 數量超過目前最大值，則更新 H
            if inliers > max_inliers:
                max_inliers = inliers
                max_H = curr_H
                max_is_corrected = is_corrected
                
        for i in range(len(max_is_corrected)):
            if (max_is_corrected[i] == 1):
                true_positive = is_point_satisfying_homography(
                    self.H,
                    self.source_points[i][0],
                    self.source_points[i][1],
                    self.target_points[i][0],
                    self.target_points[i][1],
                )
                if true_positive:
                    self.correct_matches += 1

        error = self.calculate_reprojection_error(self.source_points, self.H, max_H)
        
        self.max_inliers = max_inliers
        self.max_H = max_H
        self.max_is_corrected = max_is_corrected
        print(f"Extracted matches: {max_inliers}")
        print(f"Removed matches {self.number_of_points - max_inliers}")
        print(f"Correct matches in the extracted matches: {self.correct_matches}")
        print(f"Reprojection error: {error}")

    def plot_results(self):
        """Plot the results of matching points."""
        plot_match_points(
            self.img1_path, self.img2_path, self.source_points, self.target_points, self.max_is_corrected, self.H
        )
        
    def calculate_reprojection_error(self, points, H_gt, H_est):
        """
        計算重投影誤差 (Reprojection Error)

        Args:
            points (numpy.ndarray): 原始點集，形狀為 (N, 2)，每行為 [x, y]
            H_gt (numpy.ndarray): 真實的 Homography 矩陣，形狀為 (3, 3)
            H_est (numpy.ndarray): 估計的 Homography 矩陣，形狀為 (3, 3)

        Returns:
            float: 平均重投影誤差
        """
        # 將點擴展為齊次座標 (N, 2) -> (N, 3)
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        
        # 用 H_gt 和 H_est 映射點集
        projected_gt = H_gt @ points_homogeneous.T  # 形狀為 (3, N)
        projected_est = H_est @ points_homogeneous.T  # 形狀為 (3, N)
        
        # 正規化齊次座標 (x, y, w) -> (x/w, y/w)
        projected_gt = projected_gt[:2, :] / projected_gt[2, :]
        projected_est = projected_est[:2, :] / projected_est[2, :]
        
        # 計算歐幾里得距離的平方
        squared_errors = np.sum((projected_gt - projected_est)**2, axis=0)
        
        # 平均誤差
        mean_error = np.sqrt(np.mean(squared_errors))
        return mean_error
    def evaluate(self):
        """Evaluate the results."""
        
        precision = self.correct_matches / self.max_inliers
        recall = self.correct_matches / self.total_correct_matches
        f1 = 2 * precision * recall / (precision + recall)
        print(f"Precision: {precision} = {self.correct_matches} / {self.max_inliers}")
        print(f"Recall: {recall} = {self.correct_matches} / {self.total_correct_matches}")
        print(f"F1 Score: {f1}")

if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    img1_path = os.path.join(root, "dataset/boat/img1.pgm")
    img2_path = os.path.join(root, "dataset/boat/img3.pgm")
    ground_truth_path = os.path.join(root, "dataset/boat/H1to3p")

    lts = LTS(img1_path, img2_path, ground_truth_path)
    lts.extract_features(method="SIFT", threshold=0.8, isShow=True)
    print("-------------------")
    print("Run RANSAC")
    lts.run_ransac(threshold=3.5, max_iteration=100)
    # lts.plot_results()
    print("-------------------")
    print("Evaluate")
    lts.evaluate()