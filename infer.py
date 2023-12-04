import argparse
from typing import List
import time
import os

import torch
import numpy as np
import onnxruntime as ort
import cv2

import sys
from pathlib import Path

sys.path.append(os.path.join(
    os.getcwd(), 'extractors', 'orbslam3_features', 'lib'))
from orbslam3_features import ORBextractor
feature_extractor = ORBextractor(2000, 1.2, 8)

root_path = os.getcwd()
print('root_path: ', root_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extractor_path",
        type=str,
        required=True,
        help="Path to the ONNX model.",
    )
    return parser.parse_args()


def normalize_keypoints(keypoints, image_shape):
    x0 = image_shape[1] / 2
    y0 = image_shape[0] / 2
    scale = max(image_shape) * 0.7
    kps = np.array(keypoints)
    kps[:, 0] = (keypoints[:, 0] - x0) / scale
    kps[:, 1] = (keypoints[:, 1] - y0) / scale
    return kps


def infer(
    extractor_path=None,
):
    # Handle args
    img1 = cv2.imread(os.path.join(
        os.getcwd(), 'qualitative', 'img2', '1.jfif'), 0)
    img2 = cv2.imread(os.path.join(
        os.getcwd(), 'qualitative', 'img2', '2.jfif'), 0)

    # Load ONNX models
    providers = ["CPUExecutionProvider", "AzureExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess_options.inter_op_num_threads = 6
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    session = ort.InferenceSession(
        extractor_path, sess_options=sess_options, providers=providers
    )

    # opencv
    orb = cv2.ORB_create(1000, 1.2, 8)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # orbslam
    # kp1s_tuples, des1 = feature_extractor.detectAndCompute(img1)
    # kp2s_tuples, des2 = feature_extractor.detectAndCompute(img2)
    # kp1 = [cv2.KeyPoint(*kp) for kp in kp1s_tuples]
    # kp2 = [cv2.KeyPoint(*kp) for kp in kp2s_tuples]

    keypoints1 = np.array(
        [[kp.pt[0], kp.pt[1], kp.size / 31,
            np.deg2rad(kp.angle)] for kp in kp1],
        dtype=np.float32
    )
    keypoints2 = np.array(
        [[kp.pt[0], kp.pt[1], kp.size / 31,
            np.deg2rad(kp.angle)] for kp in kp2],
        dtype=np.float32
    )

    # boost
    # start_time = time.time()
    kps = normalize_keypoints(keypoints1, img1.shape)
    descriptors = np.unpackbits(des1, axis=1, bitorder='little')
    descriptors = descriptors * 2.0 - 1.0
    out = session.run(
        None,
        {
            "desc": descriptors.astype(np.float32),
            "kpts": kps.astype(np.float32),
        },
    )
    des1 = np.packbits((out[0] >= 0), axis=1, bitorder='little')

    kps = normalize_keypoints(keypoints2, img2.shape)
    descriptors = np.unpackbits(des2, axis=1, bitorder='little')
    descriptors = descriptors * 2.0 - 1.0
    out = session.run(
        None,
        {
            "desc": descriptors.astype(np.float32),
            "kpts": kps.astype(np.float32),
        },
    )
    des2 = np.packbits((out[0] >= 0), axis=1, bitorder='little')

    # end_time = time.time()
    # print("cost time: {:.6f} s".format(end_time - start_time))

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
    # cv2.imshow('Matching Result', matching_result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    count = 100
    start_time = time.time()
    for i in range(0, count):
        infer(**vars(args))
    end_time = time.time()
    print("cost time: {:.6f} s".format((end_time - start_time)/count))
