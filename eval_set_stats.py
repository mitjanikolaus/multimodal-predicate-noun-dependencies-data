import argparse
import os
import pickle

from PIL import Image as PIL_Image
import fiftyone
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import fiftyone.zoo as foz


from generate_semantic_eval_sets_open_images import (
    get_bounding_box_size,
    get_sharpness_of_bounding_box,
    THRESHOLD_BB_SIZE_DIFFERENCE,
    THRESHOLD_MIN_BB_SIZE,
)

# Maximum difference in bounding box sharpness for target and visual distractor
from utils import SUBJECT, show_image

THRESHOLD_BB_SHARPNESS_DIFFERENCE = 4000


def is_same_bounding_box(rel1, rel2, precision=2):
    for val_1, val_2 in zip(rel1.bounding_box, rel2.bounding_box):
        if round(round(val_1, precision) != round(val_2, precision)):
            return False
    return True


def eval_set_stats(args):
    print("Processing: ", args.input_file)
    eval_sets = pickle.load(open(args.input_file, "rb"))

    bbox_targets = []
    bbox_distractors = []
    bbox_size_diffs = []
    min_sizes = []

    good_bbs = []

    sharpness_diffs = []

    i = 0
    for key, values in eval_sets.items():
        i += 1
        if i > 10:
            break
        if len(values) > 0:
            print(f"{key}: {len(values)})")

        for sample in tqdm(values):
            for image, rel_target, rel_distractor in zip(
                [sample["img_example"], sample["img_counterexample"]],
                [
                    sample["relationship_target"],
                    sample["counterexample_relationship_target"],
                ],
                [
                    sample["relationship_visual_distractor"],
                    sample["counterexample_relationship_visual_distractor"],
                ],
            ):
                img_path = os.path.join(
                    fiftyone.config.dataset_zoo_dir, image.split("fiftyone/")[1]
                )


                size_target = get_bounding_box_size(rel_target)
                size_distractor = get_bounding_box_size(rel_distractor)
                size_diff = abs(size_target - size_distractor)
                bbox_size_diffs.append(size_diff)
                bbox_targets.append(size_target)
                bbox_distractors.append(size_distractor)

                size_min = (
                    size_target if size_target < size_distractor else size_distractor
                )



                min_sizes.append(size_min)


                if size_min > THRESHOLD_MIN_BB_SIZE:
                    if (size_diff > THRESHOLD_BB_SIZE_DIFFERENCE) and (size_min < 0.25):
                        # print(size_target)
                        # print(size_distractor)
                        # print(size_min)
                        good_bbs.append(0)

                    else:
                        img_data = PIL_Image.open(img_path)
                        sharpness_diff = abs(
                            get_sharpness_of_bounding_box(img_data, rel_target.bounding_box)
                            - get_sharpness_of_bounding_box(
                                img_data, rel_distractor.bounding_box
                            )
                        )

                        sharpness_diffs.append(sharpness_diff)

                        if sharpness_diff > THRESHOLD_BB_SHARPNESS_DIFFERENCE:
                            # TODO: use filter DETAIL!
                            show_image(img_path, [rel_target, rel_distractor])
                        else:
                            good_bbs.append(1)

                else:

                    good_bbs.append(0)

                # show_image(img_path, [rel_target, rel_distractor])

    print("size restrictions: ")
    print(np.sum(good_bbs) / len(good_bbs))

    print("targets: ", np.mean(bbox_targets))
    print("distractors: ", np.mean(bbox_distractors))

    print("diffs:")
    print("max: ", np.max(bbox_size_diffs))
    print("mean: ", np.mean(bbox_size_diffs))
    print("std: ", np.std(bbox_size_diffs))

    print(
        len(
            [
                d
                for d, min_size in zip(bbox_size_diffs, min_sizes)
                if not (
                    (d > THRESHOLD_BB_SIZE_DIFFERENCE) and min_size < 0.25
                )
            ]
        )
        / len(bbox_size_diffs)
    )


    print("sharpness diffs:")
    print("max: ", np.max(sharpness_diffs))
    print("mean: ", np.mean(sharpness_diffs))
    print("std: ", np.std(sharpness_diffs))

    print(
        len(
            [
                d
                for d in sharpness_diffs
                if d < THRESHOLD_BB_SHARPNESS_DIFFERENCE
            ]
        )
        / len(sharpness_diffs)
    )

    plt.hist(sharpness_diffs)
    plt.show()


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(type=str, dest="input_file")

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    eval_set_stats(args)
