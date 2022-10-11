import argparse
import json
import os
import pickle

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt

from utils import get_path_of_cropped_image, get_path_of_image

from visual_bert.modeling_frcnn import GeneralizedRCNN
from visual_bert.processing_image import Preprocess
from visual_bert.utils import Config, get_data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=str, required=True)
    return parser.parse_args()

OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"

OBJ_IDS = get_data(OBJ_URL)
ATTR_IDS = get_data(ATTR_URL)


# for visualizing output
def show_image_with_boxes(a):
    a = np.uint8(np.clip(a, 0, 255))
    plt.imshow(a)


def get_image_features(img_example):
    images, sizes, scales_yx = image_preprocess(img_example)
    output_dict = frcnn(
        images,
        sizes,
        scales_yx=scales_yx,
        padding="max_detections",
        max_detections=frcnn_cfg.max_detections,
        return_tensors="pt",
    )

    return output_dict


if __name__ == "__main__":
    arg_values = get_args()

    model_name_frcnn = "unc-nlp/frcnn-vg-finetuned"

    frcnn_cfg = Config.from_pretrained(model_name_frcnn)
    frcnn = GeneralizedRCNN.from_pretrained(model_name_frcnn, config=frcnn_cfg)
    image_preprocess = Preprocess(frcnn_cfg)

    eval_set = json.load(open(arg_values.eval_set, "rb"))
    image_features = {}
    image_features_cropped = {}

    for sample in tqdm(eval_set):
        img_example_path = get_path_of_image(sample["img_filename"])

        image_features[os.path.basename(img_example_path)] = get_image_features(img_example_path)

        img_example_cropped_path = get_path_of_cropped_image(img_example_path, sample["relationship_target"])

        image_features_cropped[os.path.basename(img_example_cropped_path)] = get_image_features(img_example_cropped_path)

    out_file_name = "img_features_2048.p"
    out_file_path = os.path.expanduser(
        os.path.join("~/data/multimodal_evaluation/image_features_visual_bert", out_file_name)
    )
    os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
    pickle.dump(image_features, open(out_file_path, "wb"))

    out_file_name = "img_cropped_features_2048.p"
    out_file_path_cropped = os.path.expanduser(
        os.path.join(
            "~/data/multimodal_evaluation/image_features_visual_bert", out_file_name
        )
    )
    os.makedirs(os.path.dirname(out_file_path_cropped), exist_ok=True)
    pickle.dump(image_features_cropped, open(out_file_path_cropped, "wb"))
