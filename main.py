import os
from LTS import LTS

if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    img1_path = os.path.join(root, "dataset/boat/img1.pgm")
    img2_path = os.path.join(root, "dataset/boat/img3.pgm")
    ground_truth_path = os.path.join(root, "dataset/boat/H1to3p")

    lts = LTS(img1_path, img2_path, ground_truth_path)
    lts.extract_features(method="SIFT", threshold=0.9)
    print("-------------------")
    print("Run RANSAC")
    lts.run_ransac(max_iteration=100)
    # lts.plot_results()
    print("-------------------")
    print("Evaluate")
    lts.evaluate()