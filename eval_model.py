import argparse
import base64
import csv
import os
import pickle
import sys
import numpy as np

from utils import get_target_and_distractor_sentence, get_file_name_of_cropped_image, show_sample

TSV_FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
                      "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def load_bottom_up_image_features(fname, topk=None):
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
            #TODO: remove try/except block when data is clean
            try:
                for key, shape, dtype in decode_config:
                    item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                    item[key] = item[key].reshape(shape)
                    item[key].setflags(write=False)

                data.append(item)
                if topk is not None and len(data) == topk:
                    break
            except Exception as exc:
                 print(f"Warning: Couldn't load features for {item['img_id']}: ", end="")
                 print(exc)
    print("Loaded %d images in file %s." % (len(data), fname))

    data_dict = {}
    for img_feat in data:
        data_dict[os.path.basename(img_feat["img_id"])] = img_feat

    return data_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=str)
    return parser.parse_args()


def eval_2afc(model, tokenizer, img_features, classification_score_function, args):
    """Evaluate model using 2-alternative forced choice task."""

    eval_sets = pickle.load(open(args.eval_set, "rb"))

    successes = 0
    failures = 0

    for key, set in eval_sets.items():
        print(key)
        for sample in set:
            text_target, text_distractor = get_target_and_distractor_sentence(sample)

            if "cropped" in args.img_features_path:
                key_example = get_file_name_of_cropped_image(sample["img_example"], sample["relationship_target"])
                example_features = img_features[key_example]
                key_counterexample = get_file_name_of_cropped_image(sample["img_counterexample"], sample["counterexample_relationship_target"])
                counterexample_features = img_features[key_counterexample]
            else:
                example_features = img_features[os.path.basename(sample['img_example'])]
                counterexample_features = img_features[os.path.basename(sample['img_counterexample'])]

            prob_target_match = classification_score_function(model, tokenizer, text_target, example_features, text_target)
            prob_distractor_match = classification_score_function(model, tokenizer, text_distractor, example_features, text_target)

            if prob_target_match > prob_distractor_match:
                result_example = "SUCCESS"
                successes += 1
            else:
                result_example = "FAILURE"
                failures += 1

            result_example += f" ({prob_target_match:.3f} vs. {prob_distractor_match:.3f})"

            # Counterexample: switch target and distractor sentence
            text_distractor, text_target = text_target, text_distractor

            prob_target_match = classification_score_function(model, tokenizer, text_target, counterexample_features, text_target)
            prob_distractor_match = classification_score_function(model, tokenizer, text_distractor, counterexample_features, text_target)

            if prob_target_match > prob_distractor_match:
                result_counterexample = "SUCCESS"
                successes += 1
            else:
                result_counterexample = "FAILURE"
                failures += 1

            result_counterexample += f" ({prob_target_match:.3f} vs. {prob_distractor_match:.3f})"

            # show_sample(sample, text_distractor, text_target, result_example, result_counterexample)

    print(f"Accuracy: {round(100*successes/(successes+failures), 2)}%")
