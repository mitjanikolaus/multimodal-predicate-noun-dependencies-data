import argparse
import pickle
from time import time

from fiftyone import ViewField as F
import fiftyone.zoo as foz
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image as PIL_Image, ImageFilter, ImageStat
from matplotlib.patches import Rectangle
from tqdm import tqdm

from utils import (
    SUBJECT,
    REL,
    OBJECT,
    SYNONYMS,
    SUBJECT_TUPLES,
    OBJECTS_TUPLES,
    VALID_NAMES,
    RELATIONSHIPS_TUPLES,
    RELATIONSHIPS_SPATIAL,
    log_time,
)

# Threshold for overlap of 2 bounding boxes
THRESHOLD_SAME_BOUNDING_BOX = 0.02

# Bounding boxes of objects should be at least 10% of image in width and height
THRESHOLD_MIN_BOUNDING_BOX_WIDTH = 0.1
THRESHOLD_MIN_BOUNDING_BOX_HEIGHT = 0.1

# Bounding box sharpness quotient (relative to whole image)
THRESHOLD_MIN_BOUNDING_BOX_SHARPNESS = 0.6


def get_bounding_box_size(relationship):
    bb = relationship.bounding_box
    size = bb[2] * bb[3]
    return size


def drop_synonyms(relationships, label_name):
    filtered_rels = []
    for relationship in relationships:
        label = relationship[label_name]
        label_exists = False
        for existing_rel in filtered_rels:
            if label in SYNONYMS[existing_rel[label_name]]:
                label_exists = True
                # replace existing relationship if new one is bigger
                if get_bounding_box_size(relationship) > get_bounding_box_size(
                    existing_rel
                ):
                    filtered_rels.remove(existing_rel)
                    filtered_rels.append(relationship)
                break
        if not label_exists:
            filtered_rels.append(relationship)

    return filtered_rels


def get_sum_of_bounding_box_sizes(sample):
    return (
        get_bounding_box_size(sample["relationship_target"])
        + get_bounding_box_size(sample["relationship_visual_distractor"])
        + get_bounding_box_size(sample["counterexample_relationship_target"])
        + get_bounding_box_size(sample["counterexample_relationship_visual_distractor"])
    )


def get_image_sharpness(img_data):
    image_filtered = img_data.convert("L")
    image_filtered = image_filtered.filter(ImageFilter.FIND_EDGES)

    variance_of_laplacian = ImageStat.Stat(image_filtered).var
    return variance_of_laplacian[0]


def get_sharpness_quotient_of_bounding_box(img_data, bb):
    cropped = img_data.crop(
        (
            bb[0] * img_data.width,
            bb[1] * img_data.height,
            bb[2] * img_data.width + bb[0] * img_data.width,
            bb[3] * img_data.height + bb[1] * img_data.height,
        )
    )
    sharpness_quotient = get_image_sharpness(cropped) / get_image_sharpness(img_data)
    return sharpness_quotient


def relationships_are_sharp(sample, relationships):
    # log_time(f"START Checking relationship sharpness")

    img_data = PIL_Image.open(sample.filepath)
    for relationship in relationships:
        sharpness_quotient = get_sharpness_quotient_of_bounding_box(
            img_data, relationship.bounding_box
        )
        if sharpness_quotient <= THRESHOLD_MIN_BOUNDING_BOX_SHARPNESS:
            # log_time(f"Checked relationship sharpness")
            return False
    # log_time(f"Checked relationship sharpness")
    return True


def show_image_pair(
    image_1_path,
    image_2_path,
    regions_and_attributes_1=None,
    regions_and_attributes_2=None,
):
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    img_1_data = PIL_Image.open(image_1_path)
    img_2_data = PIL_Image.open(image_2_path)

    # Transform both images to grayscale if one of them has only one channel
    if img_1_data.mode == "L" or img_2_data.mode == "L":
        img_1_data = img_1_data.convert("L")
        img_2_data = img_2_data.convert("L")

    # Make images equal size:
    if img_2_data.height > img_1_data.height:
        img_1_data_adjusted = img_1_data.crop(
            (0, 0, img_1_data.width, img_2_data.height)
        )
        img_2_data_adjusted = img_2_data
    else:
        img_1_data_adjusted = img_1_data
        img_2_data_adjusted = img_2_data.crop(
            (0, 0, img_2_data.width, img_1_data.height)
        )

    image = np.column_stack((img_1_data_adjusted, img_2_data_adjusted))

    plt.imshow(image)

    colors = ["green", "red"]
    ax = plt.gca()
    if regions_and_attributes_1:
        for relationship, color in zip(regions_and_attributes_1, colors):
            bb = relationship.bounding_box
            ax.add_patch(
                Rectangle(
                    (bb[0] * img_1_data.width, bb[1] * img_1_data.height),
                    bb[2] * img_1_data.width,
                    bb[3] * img_1_data.height,
                    fill=False,
                    edgecolor=color,
                    linewidth=3,
                )
            )
            sharpness_quotient = get_sharpness_quotient_of_bounding_box(img_1_data, bb)
            ax.text(
                bb[0] * img_1_data.width,
                bb[1] * img_1_data.height,
                f"{relationship.Label1} {relationship.label} {relationship.Label2} "
                f"(Sharpness: {sharpness_quotient:.2f})",
                bbox={"facecolor": "white", "alpha": 0.7, "pad": 10},
            )

    if regions_and_attributes_2:
        x_offset = img_1_data.width
        for relationship, color in zip(regions_and_attributes_2, colors):
            bb = relationship.bounding_box
            ax.add_patch(
                Rectangle(
                    (bb[0] * img_2_data.width + x_offset, bb[1] * img_2_data.height),
                    bb[2] * img_2_data.width,
                    bb[3] * img_2_data.height,
                    fill=False,
                    edgecolor=color,
                    linewidth=3,
                )
            )
            sharpness_quotient = get_sharpness_quotient_of_bounding_box(img_2_data, bb)
            ax.text(
                bb[0] * img_2_data.width + x_offset,
                bb[1] * img_2_data.height,
                f"{relationship.Label1} {relationship.label} {relationship.Label2} "
                f"(Sharpness: {sharpness_quotient:.2f})",
                bbox={"facecolor": "white", "alpha": 0.7, "pad": 10},
            )

    plt.tick_params(labelbottom="off", labelleft="off")
    plt.show()


def is_subj_rel_in_image(sample, subject, rel_value, rel_label):
    if sample.relationships:
        for rel in sample.relationships.detections:
            if (
                rel.Label1 in SYNONYMS[subject]
                and rel[rel_label] in SYNONYMS[rel_value]
            ):
                return True

    return False


def high_bounding_box_overlap(bb1, bb2):
    """verify that bounding boxes are not duplicate annotations (actually the (almost) the same bounding box)"""
    diffs = [abs(b1 - b2) for b1, b2 in zip(bb1, bb2)]
    if np.all([diff < THRESHOLD_SAME_BOUNDING_BOX for diff in diffs]):
        return True
    return False


def find_other_subj_with_attr(sample, relationship_target, rel_value, rel_label):
    relationships = []
    if sample.relationships:
        for relationship in sample.relationships.detections:
            if relationship.Label1 in ["Boy", "Man"] and relationship_target.Label1 in [
                "Boy",
                "Man",
            ]:
                continue
            if relationship.Label1 in [
                "Girl",
                "Woman",
            ] and relationship_target.Label1 in ["Girl", "Woman"]:
                continue
            if (
                relationship.Label1 not in SYNONYMS[relationship_target.Label1]
                and relationship[rel_label] in SYNONYMS[rel_value]
            ):
                if not high_bounding_box_overlap(
                    relationship.bounding_box, relationship_target.bounding_box
                ):
                    # For spatial relationships we require also the object to be the same
                    if (
                        relationship[rel_label] in RELATIONSHIPS_SPATIAL
                        or relationship_target[rel_label] in RELATIONSHIPS_SPATIAL
                    ):
                        if (
                            relationship[OBJECT]
                            in SYNONYMS[relationship_target[OBJECT]]
                        ):
                            relationships.append(relationship)
                    else:
                        relationships.append(relationship)

    return relationships


def find_subj_with_other_rel(sample, subject, relationship_target, rel_label):
    relationships = []
    for relationship in sample.relationships.detections:
        if (
            relationship.Label1 in SYNONYMS[subject]
            and relationship[rel_label] in VALID_NAMES[rel_label]
            and relationship[rel_label] not in SYNONYMS[relationship_target[rel_label]]
        ):
            if not high_bounding_box_overlap(
                relationship.bounding_box, relationship_target.bounding_box
            ):
                # For spatial relationships we require also the object to be the same
                if (
                    relationship[rel_label] in RELATIONSHIPS_SPATIAL
                    or relationship_target[rel_label] in RELATIONSHIPS_SPATIAL
                ):
                    if relationship[OBJECT] in SYNONYMS[relationship_target[OBJECT]]:
                        relationships.append(relationship)
                else:
                    relationships.append(relationship)

    return relationships


def get_duplicate_sample(sample, eval_set, rel_label):
    for existing_sample in eval_set:
        if (
            existing_sample["img_example"] == sample["img_example"]
            and existing_sample["img_counterexample"] == sample["img_counterexample"]
        ) or (
            existing_sample["img_example"] == sample["img_counterexample"]
            and existing_sample["img_counterexample"] == sample["img_example"]
        ):
            # For spatial relationships we require also the object to be the same
            if rel_label in RELATIONSHIPS_SPATIAL:
                if (
                        existing_sample["relationship_target"].Label1
                        == sample["relationship_target"].Label1
                        and existing_sample["relationship_target"][rel_label]
                        == sample["relationship_target"][rel_label]
                        and existing_sample["relationship_target"][OBJECT]
                        == sample["relationship_target"][OBJECT]
                ):
                    return existing_sample
            else:
                if (
                    existing_sample["relationship_target"].Label1
                    == sample["relationship_target"].Label1
                    and existing_sample["relationship_target"][rel_label]
                    == sample["relationship_target"][rel_label]
                ):
                    return existing_sample
    return None


def get_counterexample_images(
    matching_images_general, distractor_subject, relationship_target, rel_label
):
    # check that counterexample relation is in image
    is_counterexample_relation = F(SUBJECT).is_in(SYNONYMS[distractor_subject]) & F(
        rel_label
    ).is_in(SYNONYMS[relationship_target[rel_label]])
    is_big_enough = (F("bounding_box")[2] > THRESHOLD_MIN_BOUNDING_BOX_WIDTH) & (
        F("bounding_box")[3] > THRESHOLD_MIN_BOUNDING_BOX_HEIGHT
    )
    # For spatial relationships we require also the object to be the same
    if relationship_target[rel_label] in RELATIONSHIPS_SPATIAL:
        is_counterexample_relation = (
            F(SUBJECT).is_in(SYNONYMS[distractor_subject])
            & F(rel_label).is_in(SYNONYMS[relationship_target[rel_label]])
            & F(OBJECT).is_in(SYNONYMS[relationship_target[OBJECT]])
        )

    # check that target IS NOT in same image:
    is_example_relation = F(SUBJECT).is_in(SYNONYMS[relationship_target[SUBJECT]]) & F(
        rel_label
    ).is_in(SYNONYMS[relationship_target[rel_label]])
    if relationship_target[rel_label] in RELATIONSHIPS_SPATIAL:
        is_example_relation = (
            F(SUBJECT).is_in(SYNONYMS[relationship_target[SUBJECT]])
            & F(rel_label).is_in(SYNONYMS[relationship_target[rel_label]])
            & F(OBJECT).is_in(SYNONYMS[relationship_target[OBJECT]])
        )

    matching_images_counterexample = matching_images_general.match(
        (
            F("relationships.detections")
            .filter(is_counterexample_relation & is_big_enough)
            .length()
            > 0
        )
        & (F("relationships.detections").filter(is_example_relation).length() == 0)
    )
    # log_time(f"Finished computing matching samples (counterexample)")

    return matching_images_counterexample


def get_counterexample_key(relationship_target, rel_label):
    if relationship_target[rel_label] in RELATIONSHIPS_SPATIAL:
        counterexample_key = f"{relationship_target[REL]}-{relationship_target[OBJECT]}"
    else:
        counterexample_key = f"{relationship_target[rel_label]}"
    return counterexample_key


def generate_eval_sets_from_subject_tuples(
    subject_tuples, split, max_samples, file_name, check_sharpness
):
    # start_time = time()
    # log_time("Start generate eval_sets")

    eval_sets = {}

    dataset = foz.load_zoo_dataset(
        "open-images-v6",
        label_types=["relationships"],
        max_samples=max_samples,
        split=split,
        dataset_name=f"open-images-v6-{split}-{max_samples}",
    )

    for target_tuple in subject_tuples:
        print("Looking for: ", target_tuple)
        # log_time(f"Start looking for: {target_tuple}")

        eval_set = []
        counterexample_cache = {}
        target_subject, distractor_subject = target_tuple

        # Compute matching images
        is_target = F(SUBJECT).is_in(SYNONYMS[target_subject])
        is_distractor = F(SUBJECT).is_in(SYNONYMS[distractor_subject])
        is_big_enough = (F("bounding_box")[2] > THRESHOLD_MIN_BOUNDING_BOX_WIDTH) & (
            F("bounding_box")[3] > THRESHOLD_MIN_BOUNDING_BOX_HEIGHT
        )
        matching_images = dataset.match(
            (
                F("relationships.detections").filter(is_target & is_big_enough).length()
                > 0
            )
            & (
                F("relationships.detections")
                .filter(is_distractor & is_big_enough)
                .length()
                > 0
            )
        )

        # log_time(f"Computed matching samples")

        for example in tqdm(matching_images):
            # log_time(f"Start with example")

            # Look for relationships both based on REL and on OBJECT (label and Label2)
            for rel_label in [REL, OBJECT]:
                possible_relationships = [
                    rel
                    for rel in example.relationships.detections
                    if rel.Label1 in SYNONYMS[target_subject]
                    and rel[rel_label] in VALID_NAMES[rel_label]
                    and not is_subj_rel_in_image(
                        example, distractor_subject, rel[rel_label], rel_label
                    )  # check that distractor IS NOT in same image
                ]
                possible_relationships = drop_synonyms(
                    possible_relationships, rel_label
                )

                for relationship_target in possible_relationships:
                    # log_time(f"Start with relationship_target")

                    # check that visual distractor IS in image
                    rels_visual_distractor = find_subj_with_other_rel(
                        example, distractor_subject, relationship_target, rel_label
                    )
                    rels_visual_distractor = drop_synonyms(
                        rels_visual_distractor, rel_label
                    )
                    if len(rels_visual_distractor) > 0:

                        # log_time(f"Start with rel_visual_distractor")

                        if not check_sharpness or relationships_are_sharp(
                            example, [relationship_target],
                        ):

                            counterexample_key = get_counterexample_key(
                                relationship_target, rel_label
                            )

                            if counterexample_key not in counterexample_cache:
                                counterexample_cache[
                                    counterexample_key
                                ] = get_counterexample_images(
                                    matching_images,
                                    distractor_subject,
                                    relationship_target,
                                    rel_label,
                                )

                            matching_images_counterexample = counterexample_cache[
                                counterexample_key
                            ]

                            for counterexample in matching_images_counterexample:
                                # log_time(f"Start counterexample")

                                counterexample_possible_relationships = [
                                    rel
                                    for rel in counterexample.relationships.detections
                                    if rel.Label1 in SYNONYMS[distractor_subject]
                                    and rel[rel_label]
                                    in SYNONYMS[relationship_target[rel_label]]
                                ]
                                counterexample_possible_relationships = drop_synonyms(
                                    counterexample_possible_relationships, rel_label,
                                )

                                for (
                                    counterex_rel_target
                                ) in counterexample_possible_relationships:
                                    # log_time(f"Start counterex_rel_target")

                                    # check that visual distractor IS in image
                                    counterexample_rels_visual_distractor = find_subj_with_other_rel(
                                        counterexample,
                                        target_subject,
                                        counterex_rel_target,
                                        rel_label,
                                    )
                                    counterexample_rels_visual_distractor = drop_synonyms(
                                        counterexample_rels_visual_distractor, rel_label
                                    )
                                    if len(counterexample_rels_visual_distractor) > 0:
                                        for (
                                            rel_visual_distractor
                                        ) in rels_visual_distractor:
                                            for (
                                                counterex_rel_visual_distractor
                                            ) in counterexample_rels_visual_distractor:
                                                # log_time(
                                                #     f"Start counterexample_rels_visual_distractor"
                                                # )

                                                if not check_sharpness or (
                                                    relationships_are_sharp(
                                                        counterexample,
                                                        [
                                                            counterex_rel_target,
                                                            counterex_rel_visual_distractor,
                                                        ]
                                                        and relationships_are_sharp(
                                                            example,
                                                            rel_visual_distractor,
                                                        ),
                                                    ),
                                                ):

                                                    sample = {
                                                        "img_example": example.filepath,
                                                        "img_counterexample": counterexample.filepath,
                                                        "relationship_target": relationship_target,
                                                        "relationship_visual_distractor": rel_visual_distractor,
                                                        "counterexample_relationship_target": counterex_rel_target,
                                                        "counterexample_relationship_visual_distractor": counterex_rel_visual_distractor,
                                                        "rel_label": rel_label,
                                                    }
                                                    duplicate_sample = get_duplicate_sample(
                                                        sample, eval_set, rel_label
                                                    )
                                                    # log_time(f"Ready to add sample")

                                                    # Replace current sample if new one has bigger objects
                                                    if duplicate_sample is not None:
                                                        if get_sum_of_bounding_box_sizes(
                                                            sample
                                                        ) > get_sum_of_bounding_box_sizes(
                                                            duplicate_sample
                                                        ):
                                                            eval_set.remove(
                                                                duplicate_sample
                                                            )
                                                            eval_set.append(sample)
                                                            # log_time(f"Added sample")
                                                        # else:
                                                        #     log_time(f"Didn't add sample")

                                                    else:
                                                        # show_image_pair(example.filepath, counterexample.filepath, [relationship_target, rel_visual_distractor], [counterex_rel_target, counterex_rel_visual_distractor])

                                                        # Add example and counter-example
                                                        eval_set.append(sample)
                                                        # log_time(f"Added sample")

        if len(eval_set) > 0:
            eval_sets[target_tuple] = eval_set
            print("saving intermediate results..")
            pickle.dump(eval_sets, open(file_name, "wb"))
            print(
                f"\nFound {len(eval_sets[target_tuple])} examples for {target_tuple}.\n"
            )

    # log_time("End generate eval_sets", start_time)

    return eval_sets


def generate_eval_sets_from_rel_or_object_tuples(
    tuples, rel_label, split, max_samples, file_name
):
    dataset = foz.load_zoo_dataset(
        "open-images-v6",
        label_types=["relationships"],
        max_samples=max_samples,
        split=split,
        dataset_name=f"open-images-v6-{split}-{max_samples}",
    )

    eval_sets = {}

    for target_tuple in tuples:
        print("Looking for: ", target_tuple)
        eval_set = []
        target_attribute, distractor_attribute = target_tuple

        # Compute matching images
        is_target = F(rel_label).is_in(SYNONYMS[target_attribute])
        is_distractor = F(rel_label).is_in(SYNONYMS[distractor_attribute])
        is_big_enough = (F("bounding_box")[2] > THRESHOLD_MIN_BOUNDING_BOX_WIDTH) & (
            F("bounding_box")[3] > THRESHOLD_MIN_BOUNDING_BOX_HEIGHT
        )
        matching_images = dataset.match(
            (
                F("relationships.detections").filter(is_target & is_big_enough).length()
                > 0
            )
            & (
                F("relationships.detections")
                .filter(is_distractor & is_big_enough)
                .length()
                > 0
            )
        )

        for example in tqdm(matching_images):
            relationships = [
                rel
                for rel in example.relationships.detections
                if rel[rel_label] in SYNONYMS[target_attribute]
                and not is_subj_rel_in_image(
                    example, rel.Label1, distractor_attribute, rel_label
                )  # check that distractor IS NOT in same image
            ]
            relationships = drop_synonyms(relationships, SUBJECT)

            for relationship_target in relationships:
                target_subject = relationship_target.Label1

                # check that visual distractor IS in image
                rels_visual_distractor = find_other_subj_with_attr(
                    example, relationship_target, distractor_attribute, rel_label
                )
                rels_visual_distractor = drop_synonyms(rels_visual_distractor, SUBJECT)

                for rel_visual_distractor in rels_visual_distractor:
                    if relationships_are_sharp(
                        example, [relationship_target, rel_visual_distractor],
                    ):

                        # Start looking for counterexample image..
                        is_counterex_rel = F(SUBJECT).is_in(
                            SYNONYMS[target_subject]
                        ) & F(rel_label).is_in(SYNONYMS[distractor_attribute])
                        # For spatial relationships we require also the object to be the same
                        if distractor_attribute in RELATIONSHIPS_SPATIAL:
                            is_counterex_rel = (
                                F(SUBJECT).is_in(SYNONYMS[target_subject])
                                & F(rel_label).is_in(SYNONYMS[distractor_attribute])
                                & F(OBJECT).is_in(SYNONYMS[relationship_target[OBJECT]])
                            )
                        matching_images_counterexample = matching_images.match(
                            F("relationships.detections")
                            .filter(is_counterex_rel & is_big_enough)
                            .length()
                            > 0
                        )

                        # check that target IS NOT in same image:
                        matching_images_counterexample = [
                            im
                            for im in matching_images_counterexample
                            if not is_subj_rel_in_image(
                                im, target_subject, target_attribute, rel_label
                            )
                        ]

                        for counterexample in matching_images_counterexample:

                            counterexample_relationships = [
                                rel
                                for rel in counterexample.relationships.detections
                                if rel.Label1 in SYNONYMS[target_subject]
                                and rel[rel_label] in SYNONYMS[distractor_attribute]
                            ]
                            counterexample_relationships = drop_synonyms(
                                counterexample_relationships, SUBJECT,
                            )

                            for counterex_rel_target in counterexample_relationships:
                                # check that visual distractor IS in image
                                counterex_rels_visual_distractor = find_other_subj_with_attr(
                                    counterexample,
                                    counterex_rel_target,
                                    target_attribute,
                                    rel_label,
                                )
                                counterex_rels_visual_distractor = drop_synonyms(
                                    counterex_rels_visual_distractor, SUBJECT,
                                )

                                for (
                                    counterex_rel_visual_distractor
                                ) in counterex_rels_visual_distractor:
                                    if relationships_are_sharp(
                                        counterexample,
                                        [
                                            counterex_rel_target,
                                            counterex_rel_visual_distractor,
                                        ],
                                    ):
                                        sample = {
                                            "img_example": example.filepath,
                                            "img_counterexample": counterexample.filepath,
                                            "relationship_target": relationship_target,
                                            "relationship_visual_distractor": rel_visual_distractor,
                                            "counterexample_relationship_target": counterex_rel_target,
                                            "counterexample_relationship_visual_distractor": counterex_rel_visual_distractor,
                                            "rel_label": rel_label,
                                        }
                                        duplicate_sample = get_duplicate_sample(
                                            sample, eval_set, rel_label
                                        )

                                        # Replace current sample if new one has bigger objects
                                        if duplicate_sample is not None:
                                            if get_sum_of_bounding_box_sizes(
                                                sample
                                            ) > get_sum_of_bounding_box_sizes(
                                                duplicate_sample
                                            ):
                                                eval_set.remove(duplicate_sample)
                                                eval_set.append(sample)
                                        else:
                                            # show_image_pair(example.filepath, counterexample.filepath, [relationship_target, rel_visual_distractor], [counterex_rel_target, counterex_rel_visual_distractor])

                                            # Add tuple of example and counter-example
                                            eval_set.append(sample)

        if len(eval_set) > 0:
            eval_sets[target_tuple] = eval_set
            print("saving intermediate results..")
            pickle.dump(eval_sets, open(file_name, "wb"))
            print(
                f"Found {len(eval_sets[target_tuple])} examples for {target_tuple}.\n\n"
            )

    return eval_sets


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--eval-set",
        type=str,
        required=True,
        choices=["subject_tuples", "relationship_tuples", "object_tuples"],
    )
    argparser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "validation", "test", None],
    )
    argparser.add_argument(
        "--max-samples", type=int, default=None,
    )
    argparser.add_argument("--check-sharpness", default=False, action="store_true")

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.eval_set == "subject_tuples":
        file_name = f"results/subject-{args.split}-{args.max_samples}.p"
        eval_sets_based_on_subjects = generate_eval_sets_from_subject_tuples(
            SUBJECT_TUPLES,
            args.split,
            args.max_samples,
            file_name,
            args.check_sharpness,
        )
        pickle.dump(eval_sets_based_on_subjects, open(file_name, "wb"))

    elif args.eval_set == "object_tuples":
        file_name = f"results/object-{args.split}-{args.max_samples}.p"
        eval_sets = generate_eval_sets_from_rel_or_object_tuples(
            OBJECTS_TUPLES, OBJECT, args.split, args.max_samples, file_name
        )
        pickle.dump(eval_sets, open(file_name, "wb"))

    elif args.eval_set == "relationship_tuples":
        file_name = f"results/rel-{args.split}-{args.max_samples}.p"
        eval_sets = generate_eval_sets_from_rel_or_object_tuples(
            RELATIONSHIPS_TUPLES, REL, args.split, args.max_samples, file_name
        )
        pickle.dump(eval_sets, open(file_name, "wb"))
