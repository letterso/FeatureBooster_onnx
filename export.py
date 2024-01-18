from pathlib import Path
import argparse
from typing import List
import numpy as np
import yaml
import cv2
import torch
import onnx
from onnxsim import simplify, model_info

from featurebooster import FeatureBooster

import sys
sys.path.append(str(Path.cwd()/'extractors'/'orbslam3_features'/'lib'))
from orbslam3_features import ORBextractor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device == 'cuda:0':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--descriptor",
        type=str,
        default="ORB+Boost-B",
        choices=["ORB+Boost-B", "SuperPoint+Boost-B", "SuperPoint+Boost-F"],
        required=False,
        help="Type of feature extractor. Supported extractors are 'ORB+Boost-B', 'SuperPoint+Boost-B', 'SuperPoint+Boost-F'. Defaults to 'ORB+Boost-B'.",
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
    img_path="qualitative/img2/1.jfif",
):
    extractor_path = str(Path.cwd()/'weights' / '{}.onnx'.format(descriptor))

    # orb
    feature_extractor = ORBextractor(3000, 1.2, 8)
    image = cv2.imread(img_path)

    # set mode
    with (Path.cwd()/'config.yam').open('r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config[descriptor])

    # load the model
    feature_booster = FeatureBooster(config[descriptor])
    feature_booster.load_state_dict(torch.load(str(Path.cwd()/'models'/'{}.pth'.format(descriptor))))
    feature_booster.to(device).eval()

    # extract kpts
    kps_tuples, descriptors = feature_extractor.detectAndCompute(image)
    # convert keypoints
    keypoints = [cv2.KeyPoint(*kp) for kp in kps_tuples]
    keypoints = np.array(
        [[kp.pt[0], kp.pt[1], kp.size / 31,
            np.deg2rad(kp.angle)] for kp in keypoints],
        dtype=np.float32
    )
    # ONNX Export
    kps = normalize_keypoints(keypoints, image.shape)
    kps = torch.from_numpy(kps.astype(np.float32)).to(device)
    descriptors = np.unpackbits(descriptors, axis=1, bitorder='little')
    descriptors = descriptors * 2.0 - 1.0
    descriptors = torch.from_numpy(descriptors.astype(np.float32)).to(device)
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

    mode_ori = onnx.load(extractor_path)

    # 优化
    model_simp, check_ok = simplify(mode_ori)
    if check_ok:
        print("Finish! Here is the difference:")
        model_info.print_simplifying_info(mode_ori, model_simp)
    else:
        print(
            'Check failed. Please be careful to use the simplified model, or try specifying "--skip-fuse-bn" or "--skip-optimization" (run "onnxsim -h" for details).'
        )
        print("Here is the difference after simplification:")
        model_info.print_simplifying_info(mode_ori, model_simp)
        return

    onnx.save(model_simp, str(Path.cwd()/'weights' /
              '{}_opt.onnx'.format(descriptor)))

if __name__ == "__main__":
    args = parse_args()
    export_onnx(**vars(args))
