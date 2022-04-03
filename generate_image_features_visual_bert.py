import argparse
import os
import pickle

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt

from filter_eval_set import get_local_image_path
from utils import get_path_of_cropped_image

from PIL import Image

from visual_bert.modeling_frcnn import GeneralizedRCNN
from visual_bert.processing_image import Preprocess
from visual_bert.utils import Config, get_data
from visual_bert.visualizing_image import SingleImageViz


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


def get_image_features(img_example, img_counterexample):

    # frcnn_visualizer = SingleImageViz(img_example, id2obj=OBJ_IDS, id2attr=ATTR_IDS)
    images, sizes, scales_yx = image_preprocess(img_example)
    output_dict = frcnn(
        images,
        sizes,
        scales_yx=scales_yx,
        padding="max_detections",
        max_detections=frcnn_cfg.max_detections,
        return_tensors="pt",
    )

    images_2, sizes_2, scales_yx_2 = image_preprocess(img_counterexample)
    output_dict_2 = frcnn(
        images_2,
        sizes_2,
        scales_yx=scales_yx_2,
        padding="max_detections",
        max_detections=frcnn_cfg.max_detections,
        return_tensors="pt",
    )

    # # add boxes and labels to the image
    # frcnn_visualizer.draw_boxes(
    #     output_dict.get("boxes"),
    #     output_dict.pop("obj_ids"),
    #     output_dict.pop("obj_probs"),
    #     output_dict.pop("attr_ids"),
    #     output_dict.pop("attr_probs"),
    # )
    # show_image_with_boxes(frcnn_visualizer._get_buffer())

    return output_dict, output_dict_2


if __name__ == "__main__":
    arg_values = get_args()

    model_name_frcnn = "unc-nlp/frcnn-vg-finetuned"

    frcnn_cfg = Config.from_pretrained(model_name_frcnn)
    frcnn = GeneralizedRCNN.from_pretrained(model_name_frcnn, config=frcnn_cfg)
    image_preprocess = Preprocess(frcnn_cfg)

    eval_sets = pickle.load(open(arg_values.eval_set, "rb"))
    image_features = {}
    image_features_cropped = {}

    for key, set in eval_sets.items():
        print(key)
        for sample in tqdm(set):
            img_example_path = get_local_image_path(sample["img_example"])
            img_counterexample_path = get_local_image_path(sample["img_counterexample"])

            (
                image_features[img_example_path],
                image_features[img_counterexample_path],
            ) = get_image_features(img_example_path, img_counterexample_path)

            img_example_cropped_path = get_path_of_cropped_image(img_example_path, sample["relationship_target"])

            img_counterexample_cropped_path = get_path_of_cropped_image(img_counterexample_path, sample["counterexample_relationship_target"])
            (
                image_features_cropped[img_example_cropped_path],
                image_features_cropped[img_counterexample_cropped_path],
            ) = get_image_features(img_example_cropped_path, img_counterexample_cropped_path)

    out_file_name = os.path.basename(arg_values.eval_set)
    out_file_path = os.path.expanduser(
        os.path.join("~/data/multimodal_evaluation/image_features_visual_bert", out_file_name)
    )
    os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
    pickle.dump(image_features, open(out_file_path, "wb"))

    out_file_name = os.path.basename(arg_values.eval_set)
    out_file_path_cropped = os.path.expanduser(
        os.path.join(
            "~/data/multimodal_evaluation/image_features_visual_bert_cropped", out_file_name
        )
    )
    os.makedirs(os.path.dirname(out_file_path_cropped), exist_ok=True)
    pickle.dump(image_features_cropped, open(out_file_path_cropped, "wb"))
