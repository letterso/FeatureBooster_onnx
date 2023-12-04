import cv2
import numpy as np
import yaml
import torch
from featurebooster import FeatureBooster
import time
import os

import sys
from pathlib import Path


# sys.path.append(os.path.join(
#     os.getcwd(), 'extractors', 'orbslam2_features', 'lib'))
# from orbslam2_features import ORBextractor
# feature_extractor = ORBextractor(1000, 1.2, 2)

sys.path.append(os.path.join(
    os.getcwd(), 'extractors', 'orbslam3_features', 'lib'))
from orbslam3_features import ORBextractor
feature_extractor = ORBextractor(1000, 1.2, 8)


def normalize_keypoints(keypoints, image_shape):
    x0 = image_shape[1] / 2
    y0 = image_shape[0] / 2
    scale = max(image_shape) * 0.7
    kps = np.array(keypoints)
    kps[:, 0] = (keypoints[:, 0] - x0) / scale
    kps[:, 1] = (keypoints[:, 1] - y0) / scale
    return kps


# 加载图像
img1 = cv2.imread(os.path.join(
    os.getcwd(), 'qualitative', 'img2', '1.jfif'), 0)
img2 = cv2.imread(os.path.join(
    os.getcwd(), 'qualitative', 'img2', '2.jfif'), 0)


# opencv
orb = cv2.ORB_create(1000, 1.2, 8)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# orbslam
# kp1s_tuples, des1 = feature_extractor.detectAndCompute(img1)
# kp2s_tuples, des2 = feature_extractor.detectAndCompute(img2)
# kp1 = [cv2.KeyPoint(*kp) for kp in kp1s_tuples]
# kp2 = [cv2.KeyPoint(*kp) for kp in kp2s_tuples]

# boost
# load json config file
config_file = Path(__file__).parent / "config.yaml"
with open(str(config_file), 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
print(config['ORB+Boost-B'])

feature_booster = FeatureBooster(config['ORB+Boost-B'])
feature_booster.eval()
# load the model
model_path = Path(__file__).parent / str("models/" + 'ORB+Boost-B' + ".pth")
print(model_path)
feature_booster.load_state_dict(torch.load(model_path))

# convert keypoints
start_time = time.time()
keypoints1 = np.array(
    [[kp.pt[0], kp.pt[1], kp.size / 31, np.deg2rad(kp.angle)] for kp in kp1],
    dtype=np.float32
)
kps = normalize_keypoints(keypoints1, img1.shape)
kps = torch.from_numpy(kps.astype(np.float32))
descriptors = np.unpackbits(des1, axis=1, bitorder='little')
descriptors = descriptors * 2.0 - 1.0
descriptors = torch.from_numpy(descriptors.astype(np.float32))
out = feature_booster(descriptors, kps)
out = (out >= 0).cpu().detach().numpy()
des1 = np.packbits(out, axis=1, bitorder='little')

keypoints2 = np.array(
    [[kp.pt[0], kp.pt[1], kp.size / 31, np.deg2rad(kp.angle)] for kp in kp2],
    dtype=np.float32
)
kps = normalize_keypoints(keypoints2, img2.shape)
kps = torch.from_numpy(kps.astype(np.float32))
descriptors = np.unpackbits(des2, axis=1, bitorder='little')
descriptors = descriptors * 2.0 - 1.0
descriptors = torch.from_numpy(descriptors.astype(np.float32))
out = feature_booster(descriptors, kps)
out = (out >= 0).cpu().detach().numpy()
des2 = np.packbits(out, axis=1, bitorder='little')

end_time = time.time()
print("cost time: {:.6f} ms".format((end_time - start_time)*1000))

# 创建BFMatcher对象
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# 使用knnMatch进行特征匹配
matches = bf.knnMatch(des1, des2, k=2)

# 应用筛选条件
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 绘制匹配结果
matching_result = cv2.drawMatches(
    img1, kp1, img2, kp2, good_matches, None, flags=2)

# 显示匹配结果
# cv2.imwrite(os.path.join(os.getcwd(), 'assets', 'match.png'), matching_result)
cv2.imshow('Matching Result', matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
