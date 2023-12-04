import argparse
from typing import List
import numpy as np
import yaml
import cv2
import torch

from featurebooster import FeatureBooster

import sys
from pathlib import Path
orb_path = Path(__file__).parent / "extractors/orbslam2_features/lib"
sys.path.append(str(orb_path))
from orbslam2_features import ORBextractor

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--descriptor",
        type=str,
        default="ORB+Boost-B",
        choices=["ORB+Boost-B", "SuperPoint+Boost-B","SuperPoint+Boost-F"],
        required=False,
        help="Type of feature extractor. Supported extractors are 'ORB+Boost-B', 'SuperPoint+Boost-B', 'SuperPoint+Boost-F'. Defaults to 'ORB+Boost-B'.",
    )
    parser.add_argument(
        "--extractor_path",
        type=str,
        default=None,
        required=True,
        help="Path to save the feature extractor ONNX model.",
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

def export_onnx(
    descriptor="ORB+Boost-B",
    extractor_path=None,
    img_path="assets/sacre_coeur1.jpg",
):
    feature_extractor = ORBextractor(3000, 1.2, 8)
    image = cv2.imread(img_path)

    # set mode
    config_file = Path(__file__).parent / "config.yaml"
    with open(str(config_file), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config[descriptor])

    # load the model
    feature_booster = FeatureBooster(config[descriptor])
    feature_booster.eval()
    model_path = Path(__file__).parent / str("models/" + descriptor + ".pth")
    print(model_path)
    feature_booster.load_state_dict(torch.load(model_path))

    # extract kpts
    kps_tuples, descriptors = feature_extractor.detectAndCompute(image)
    # convert keypoints 
    keypoints = [cv2.KeyPoint(*kp) for kp in kps_tuples]
    keypoints = np.array(
        [[kp.pt[0], kp.pt[1], kp.size / 31, np.deg2rad(kp.angle)] for kp in keypoints], 
        dtype=np.float32
    )
    # ONNX Export
    kps = normalize_keypoints(keypoints, image.shape)
    kps = torch.from_numpy(kps.astype(np.float32))
    descriptors = np.unpackbits(descriptors, axis=1, bitorder='little')
    descriptors = descriptors * 2.0 - 1.0
    descriptors = torch.from_numpy(descriptors.astype(np.float32))
    torch.onnx.export(
        feature_booster,
        (descriptors, kps),
        extractor_path,
        input_names=["desc", "kpts"],
        output_names=["desc_boost"],
        opset_version=17,
        dynamic_axes={
            "desc": {0: "num_desc"},
            "kpts": {0: "num_kpts"},
            "desc_boost": {0: "num_desc_boost"},
        },
    )


if __name__ == "__main__":
    args = parse_args()
    export_onnx(**vars(args))
