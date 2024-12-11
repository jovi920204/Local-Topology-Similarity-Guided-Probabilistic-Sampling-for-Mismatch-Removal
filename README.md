# Local Topology Similarity Guided Probabilistic Sampling for Mismatch Removal
This repo is not an offical code

## Dataset
[Mikolajczyk Dataset](https://www.robots.ox.ac.uk/~vgg/research/affine/)
- 下載 `boat.tar.gz` 的檔案，並解壓縮至 `dataset\` 資料夾底下

## Install Python Package
```bash
pip install -r requirments.txt
```

## Usage
```bash
python main.py
```

## Evaluation
1.	Precision:
	- 計算方式：precision = correct_matches / extracted_matches
	- correct_matches: 正確匹配的數量 (從方法中提取出來的匹配中真正正確的數量)
	- extracted_matches: 所有提取出的匹配數量 (無論正確與否)
2.	Recall:
    - 計算方式：recall = correct_matches / total_correct_matches
	- correct_matches: 同上，正確匹配的數量。
	- total_correct_matches: 總共的正確匹配數量 (包括未提取的部分)。

## Program Explanation
- `class LTS()`
    1. `LTS()` 輸入兩張圖片的影像，以及 ground truth H 
    2. `extract_features()` 提取特徵點（可使用 SIFT 或 ORB），並使用 BFMatcher 找出匹配點（match points）
    3. `run_ransac()` 實作 LTS 演算法，並使用 RANSAC 取樣特徵點，最終過濾 mismatces
    4. `evaluate()` 計算 precision, recall, f-score
- `extract_feature()`
    - `method`: 可以使用 `SIFT` 或是 `ORB`，該程式碼放在 FeatureExtraction.py 中
    - `threshold`: 在使用 BFMatcher 的 knnMatch 時，會回傳最近鄰居（m），以及次近鄰居（n），若 m 的距離顯著小於 n 的距離（利用 threshold 控制顯著比例），則可以挑選出更可能為正確的 match point
        - 因此可使用此參數設定 outlier rate
        - 根據論文，outlier rate < 0.7，LTS F-score 皆有 0.9 以上
- `run_ransac()`
    - `max_iteration`: RANSAC 迭代的最大次數
    - 依照論文的方法，實現每個步驟
        - Contruct topology network
        - Calculate the mismatching probability
        - Probability guided sampling
    - `max_inliers`: 透過 LTS 過濾出來的 correct matches 數量
    - `max_H`: 估計出最好的 Homography Matrix
    - `max_is_corrected`: 透過 LTS 過濾出來的 correct matches index
    - `correct_matches`: 過濾出來的 inlier，實際也為 inlier 的數量

- `evaluate`
    ```python
    precision = self.correct_matches / self.max_inliers
    recall = self.correct_matches / self.total_correct_matches
    f1 = 2 * precision * recall / (precision + recall)
    ```