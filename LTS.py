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
        self.true_inlier = None
        self.true_outlier = None
        self.number_of_points = None # total number of matching points
        self.total_extracted_matches = 0
        self.extracted_correct_matches = 0
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

    def extract_features(self, method="SIFT", threshold=0.2834):
        """Extract features using SIFT or ORB."""
        if method == "SIFT":
            self.source_points, self.target_points, self.true_inlier, self.true_outlier = SIFT(
                self.img1_path, self.img2_path, self.H, threshold
            )
        elif method == "ORB":
            self.source_points, self.target_points = ORB(
                self.img1_path, self.img2_path, self.H, threshold
            )
        else:
            raise ValueError("Unsupported feature extraction method.")
        self.number_of_points = self.source_points.shape[0]

    def run_ransac(self, threshold=0.16, max_iteration=100):
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
                    tolerance=threshold,
                )
                if check:
                    is_corrected[i] = 1
                    inliers += 1
                    
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
                    tolerance=threshold,
                )
                if true_positive:
                    self.extracted_correct_matches += 1

        self.max_inliers = max_inliers
        self.max_H = max_H
        self.max_is_corrected = max_is_corrected
        print(f"LTS Number of max inliers: {max_inliers}")
        print(f"LTS Number of outliers: {self.number_of_points - max_inliers}")

    def plot_results(self):
        """Plot the results of matching points."""
        plot_match_points(
            self.img1_path, self.img2_path, self.source_points, self.target_points, self.max_is_corrected, self.H
        )
        
    def evaluate(self):
        """Evaluate the results."""
        
        precision = self.extracted_correct_matches / self.max_inliers
        recall = self.extracted_correct_matches / self.number_of_points
        f1 = 2 * precision * recall / (precision + recall)
        print(f"Precision: {precision} = {self.extracted_correct_matches} / {self.max_inliers}")
        print(f"Recall: {recall} = {self.extracted_correct_matches} / {self.number_of_points}")
        print(f"F1 Score: {f1}")
        

if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    img1_path = os.path.join(root, "dataset/boat/img1.pgm")
    img2_path = os.path.join(root, "dataset/boat/img3.pgm")
    ground_truth_path = os.path.join(root, "dataset/boat/H1to3p")

    lts = LTS(img1_path, img2_path, ground_truth_path)
    lts.extract_features(method="SIFT", threshold=0.8)
    print("-------------------")
    print("Run RANSAC")
    lts.run_ransac(threshold=3.5, max_iteration=100)
    # lts.plot_results()
    print("-------------------")
    print("Evaluate")
    lts.evaluate()