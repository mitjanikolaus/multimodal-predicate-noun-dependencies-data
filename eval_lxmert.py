import argparse
import base64
import csv
import os
import pickle
import sys
import time

import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, VisualBertModel, VisualBertForPreTraining, LxmertTokenizer, LxmertForPreTraining

from eval_model import eval_2afc
from utils import get_target_and_distractor_sentence, get_local_image_path, show_sample


TSV_FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
                      "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def load_tsv_image_features(fname, topk=None):
    """Load object features from tsv file.
    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    csv.field_size_limit(sys.maxsize)

    data = []
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, TSV_FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])

            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes,), np.int64),
                ('objects_conf', (boxes,), np.float32),
                ('attrs_id', (boxes,), np.int64),
                ('attrs_conf', (boxes,), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    print("Loaded %d images in file %s." % (len(data), fname))

    data_dict = {}
    for img_feat in data:
        data_dict[os.path.basename(img_feat["img_id"])] = img_feat

    return data_dict


def get_classification_score(model, tokenizer, test_sentence, visual_features):
    inputs = tokenizer(test_sentence, return_tensors="pt")

    decoded_sequence = tokenizer.decode(inputs["input_ids"][0])
    print(decoded_sequence)

    visual_embeds = torch.tensor(visual_features['features']).unsqueeze(0)

    # Normalize the boxes (to 0 ~ 1)
    image_height = visual_features["img_h"]
    image_width = visual_features["img_w"]
    boxes = torch.tensor(visual_features['boxes'])

    boxes[:, (0, 2)] /= image_width
    boxes[:, (1, 3)] /= image_height
    np.testing.assert_array_less(boxes, 1 + 1e-5)
    np.testing.assert_array_less(-boxes, 0 + 1e-5)

    boxes = boxes.unsqueeze(0)

    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

    inputs.update({
        "visual_feats": visual_embeds,
        "visual_pos": boxes,
        "visual_attention_mask": visual_attention_mask
    })

    outputs = model(**inputs)

    cross_relationship_score = outputs.cross_relationship_score

    # Apply softmax and return probability for match:
    softmaxed = F.softmax(cross_relationship_score[0], dim=0)
    return softmaxed[0].data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=str, required=True)
    parser.add_argument("--img-features-path", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')
    model = LxmertForPreTraining.from_pretrained('unc-nlp/lxmert-base-uncased')

    img_features = load_tsv_image_features(args.img_features_path)

    eval_2afc(model, tokenizer, img_features, get_classification_score, args)
