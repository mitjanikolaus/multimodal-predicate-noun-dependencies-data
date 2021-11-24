import argparse
import pickle
import threading
from multiprocessing.pool import Pool

from fiftyone import ViewField as F
import fiftyone.zoo as foz

from PIL import Image as PIL_Image, ImageFilter, ImageStat
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
    BOUNDING_BOX,
    IMAGE_RELATIONSHIPS,
    high_bounding_box_overlap, sample_to_dict,
)


# Bounding boxes of objects should be at least 25% of image in width and height
THRESHOLD_MIN_BB_SIZE = 0.25 * 0.25

# Maximum difference in bounding box area for target and visual distractor
THRESHOLD_BB_SIZE_DIFFERENCE = 0.5

# We allow high difference in bounding box size if the smaller bounding box is big enough (at least 50% in width/height)
MIN_BOUNDING_BOX_SIZE_IF_HIGH_DIFFERENCE = 0.5 * 0.5

# Bounding box sharpness quotient (relative to whole image)
THRESHOLD_MIN_BB_SHARPNESS = 0.6


def get_bounding_box_size(relationship):
    bb = relationship["bounding_box"]
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


def get_sharpness_of_bounding_box(img_data, bb):
    cropped = img_data.crop(
        (
            bb[0] * img_data.width,
            bb[1] * img_data.height,
            bb[2] * img_data.width + bb[0] * img_data.width,
            bb[3] * img_data.height + bb[1] * img_data.height,
        )
    )
    return get_image_sharpness(cropped)


def get_sharpness_quotient_of_bounding_box(img_data, bb):
    sharpness_quotient = get_sharpness_of_bounding_box(
        img_data, bb
    ) / get_image_sharpness(img_data)
    return sharpness_quotient


def relationships_are_sharp(sample, rels):
    img_data = PIL_Image.open(sample.filepath)
    for relationship in rels:
        sharpness_quotient = get_sharpness_quotient_of_bounding_box(
            img_data, relationship.bounding_box
        )
        if sharpness_quotient <= THRESHOLD_MIN_BB_SHARPNESS:
            return False
    return True


def is_subj_rel_in_image(sample, subject, rel_value, rel_label):
    for rel in sample["relationships"]:
        if (
            rel[SUBJECT] in SYNONYMS[subject]
            and rel[rel_label] in SYNONYMS[rel_value]
        ):
            return True

    return False


def find_other_subj_with_attr(sample, relationship_target, rel_value, rel_label):
    rels = []
    if sample.relationships:
        for relationship in sample.relationships.detections:
            if relationship[SUBJECT] in ["Boy", "Man"] and relationship_target[
                SUBJECT
            ] in [
                "Boy",
                "Man",
            ]:
                continue
            if (
                relationship[SUBJECT]
                in [
                    "Girl",
                    "Woman",
                ]
                and relationship_target[SUBJECT] in ["Girl", "Woman"]
            ):
                continue
            if (
                relationship[SUBJECT] not in SYNONYMS[relationship_target[SUBJECT]]
                and relationship[rel_label] in SYNONYMS[rel_value]
            ):
                # For spatial relationships we require also the object to be the same
                if (
                    relationship[rel_label] not in RELATIONSHIPS_SPATIAL
                    and relationship_target[rel_label] not in RELATIONSHIPS_SPATIAL
                ) or relationship[OBJECT] in SYNONYMS[relationship_target[OBJECT]]:
                    # Make sure the bounding box is big enough
                    if get_bounding_box_size(relationship) > THRESHOLD_MIN_BB_SIZE:
                        # Make sure the bounding boxes are not actually the same object
                        if not high_bounding_box_overlap(
                            relationship.bounding_box, relationship_target.bounding_box
                        ):
                            # Make sure the size difference to the target object is not too big
                            size_target = get_bounding_box_size(relationship_target)
                            size_distractor = get_bounding_box_size(relationship)
                            size_diff = abs(size_target - size_distractor)
                            size_min = (
                                size_target
                                if size_target < size_distractor
                                else size_distractor
                            )
                            if (size_diff < THRESHOLD_BB_SIZE_DIFFERENCE) or (
                                size_min > MIN_BOUNDING_BOX_SIZE_IF_HIGH_DIFFERENCE
                            ):
                                rels.append(relationship)

    return rels


def find_subj_with_other_rel(sample, subject, relationship_target, rel_label):
    rels = []
    for relationship in sample["relationships"]:
        # Check for correct labels
        if (
            relationship[SUBJECT] in SYNONYMS[subject]
            and relationship[rel_label] in VALID_NAMES[rel_label]
            and relationship[rel_label] not in SYNONYMS[relationship_target[rel_label]]
        ):
            # For spatial relationships we require also the object to be the same
            if (
                relationship[rel_label] not in RELATIONSHIPS_SPATIAL
                and relationship_target[rel_label] not in RELATIONSHIPS_SPATIAL
            ) or relationship[OBJECT] in SYNONYMS[relationship_target[OBJECT]]:
                # Make sure the bounding box is big enough
                if get_bounding_box_size(relationship) > THRESHOLD_MIN_BB_SIZE:
                    # Make sure the bounding boxes are not actually the same object
                    if not high_bounding_box_overlap(
                        relationship["bounding_box"], relationship_target["bounding_box"]
                    ):
                        # Make sure the size difference to the target object is not too big
                        size_target = get_bounding_box_size(relationship_target)
                        size_distractor = get_bounding_box_size(relationship)
                        size_diff = abs(size_target - size_distractor)
                        size_min = (
                            size_target
                            if size_target < size_distractor
                            else size_distractor
                        )
                        if (size_diff < THRESHOLD_BB_SIZE_DIFFERENCE) or (
                            size_min > MIN_BOUNDING_BOX_SIZE_IF_HIGH_DIFFERENCE
                        ):
                            rels.append(relationship)
    return rels


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
            if sample["relationship_target"][rel_label] in RELATIONSHIPS_SPATIAL:
                if (
                    existing_sample["relationship_target"][SUBJECT]
                    == sample["relationship_target"][SUBJECT]
                    and existing_sample["relationship_target"][rel_label]
                    == sample["relationship_target"][rel_label]
                    and existing_sample["relationship_target"][OBJECT]
                    == sample["relationship_target"][OBJECT]
                ):
                    return existing_sample
            else:
                if (
                    existing_sample["relationship_target"][SUBJECT]
                    == sample["relationship_target"][SUBJECT]
                    and existing_sample["relationship_target"][rel_label]
                    == sample["relationship_target"][rel_label]
                ):
                    return existing_sample
    return None


def get_counterexample_images_subj(
    matching_images, distractor_subject, relationship_target, rel_label
):
    matching_images_counterexample = []
    for i in matching_images:
        contains_counterexample = False
        contains_example = False
        for r in i["relationships"]:
            if r[SUBJECT] in SYNONYMS[distractor_subject] and r[rel_label] in SYNONYMS[relationship_target[rel_label]]:
                if r[BOUNDING_BOX][2] * r[BOUNDING_BOX][3] > THRESHOLD_MIN_BB_SIZE:
                    if relationship_target[rel_label] in RELATIONSHIPS_SPATIAL:
                        if r[OBJECT] in SYNONYMS[relationship_target[OBJECT]]:
                            contains_example = True
                    else:
                        contains_counterexample = True
            if r[SUBJECT] in SYNONYMS[relationship_target[SUBJECT]] and r[rel_label] in SYNONYMS[relationship_target[rel_label]]:
                if relationship_target[rel_label] in RELATIONSHIPS_SPATIAL:
                    if r[OBJECT] in SYNONYMS[relationship_target[OBJECT]]:
                        contains_example = True
                else:
                    contains_example = True
        if contains_counterexample and not contains_example:
            matching_images_counterexample.append(i)

    return matching_images_counterexample


def get_counterexample_images_attr(
    matching_images,
    distractor_attribute,
    relationship_target,
    rel_label,
):
    # check that counterexample relation is in image
    is_counterexample_relation = F(SUBJECT).is_in(
        SYNONYMS[relationship_target[SUBJECT]]
    ) & F(rel_label).is_in(SYNONYMS[distractor_attribute])
    is_big_enough = F(BOUNDING_BOX)[2] * F(BOUNDING_BOX)[3] > THRESHOLD_MIN_BB_SIZE
    # For spatial relationships we require also the object to be the same
    if relationship_target[rel_label] in RELATIONSHIPS_SPATIAL:
        is_counterexample_relation = (
            F(SUBJECT).is_in(SYNONYMS[relationship_target[SUBJECT]])
            & F(rel_label).is_in(SYNONYMS[distractor_attribute])
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

    matching_images_counterexample = matching_images.match(
        (
            F(IMAGE_RELATIONSHIPS)
            .filter(is_counterexample_relation & is_big_enough)
            .length()
            > 0
        )
        & (F(IMAGE_RELATIONSHIPS).filter(is_example_relation).length() == 0)
    )
    return matching_images_counterexample


def get_counterexample_key(relationship_target, rel_label):
    if relationship_target[rel_label] in RELATIONSHIPS_SPATIAL:
        counterexample_key = f"{relationship_target[REL]}-{relationship_target[OBJECT]}"
    else:
        counterexample_key = f"{relationship_target[rel_label]}"
    return counterexample_key


def process_sample_subj(
    example,
    matching_images,
    target_subject,
    distractor_subject,
    check_sharpness,
):
    counterexample_cache = {}
    samples = []
    # Look for relationships both based on REL and on OBJECT (label and Label2)
    for rel_label in [REL, OBJECT]:
        candidate_relationships = [
            rel
            for rel in example["relationships"]
            if rel[SUBJECT] in SYNONYMS[target_subject]
            and rel[rel_label] in VALID_NAMES[rel_label]
            and get_bounding_box_size(rel) > THRESHOLD_MIN_BB_SIZE
            and not is_subj_rel_in_image(
                example, distractor_subject, rel[rel_label], rel_label
            )  # check that distractor IS NOT in same image
        ]
        candidate_relationships = drop_synonyms(candidate_relationships, rel_label)

        for relationship_target in candidate_relationships:
            # check that visual distractor IS in image
            rels_visual_distractor = find_subj_with_other_rel(
                example, distractor_subject, relationship_target, rel_label
            )
            rels_visual_distractor = drop_synonyms(rels_visual_distractor, rel_label)
            if len(rels_visual_distractor) > 0:

                if not check_sharpness or relationships_are_sharp(
                    example,
                    [relationship_target],
                ):
                    # Start looking for counterexample image..
                    counterexample_key = get_counterexample_key(
                        relationship_target, rel_label
                    )

                    if counterexample_key not in counterexample_cache:
                        counterex_images = get_counterexample_images_subj(
                            matching_images,
                            distractor_subject,
                            relationship_target,
                            rel_label,
                        )
                        counterexample_cache[counterexample_key] = counterex_images

                    matching_images_counterexample = counterexample_cache[
                        counterexample_key
                    ]

                    for counterexample in matching_images_counterexample:

                        counterexample_relationships = [
                            rel
                            for rel in counterexample["relationships"]
                            if rel[SUBJECT] in SYNONYMS[distractor_subject]
                            and rel[rel_label]
                            in SYNONYMS[relationship_target[rel_label]]
                            and get_bounding_box_size(rel) > THRESHOLD_MIN_BB_SIZE
                        ]
                        counterexample_relationships = drop_synonyms(
                            counterexample_relationships,
                            rel_label,
                        )

                        for counterex_rel_target in counterexample_relationships:

                            # check that visual distractor IS in image
                            counterex_rels_visual_distractor = find_subj_with_other_rel(
                                counterexample,
                                target_subject,
                                counterex_rel_target,
                                rel_label,
                            )
                            counterex_rels_visual_distractor = drop_synonyms(
                                counterex_rels_visual_distractor, rel_label
                            )
                            if len(counterex_rels_visual_distractor) > 0:
                                for rel_visual_distractor in rels_visual_distractor:
                                    for (
                                        counterex_rel_visual_distractor
                                    ) in counterex_rels_visual_distractor:

                                        if not check_sharpness or (
                                            relationships_are_sharp(
                                                counterexample,
                                                [
                                                    counterex_rel_target,
                                                    counterex_rel_visual_distractor,
                                                ],
                                            )
                                            and relationships_are_sharp(
                                                example,
                                                [rel_visual_distractor],
                                            ),
                                        ):

                                            sample = {
                                                "img_example": example["filepath"],
                                                "img_counterexample": counterexample["filepath"],
                                                "relationship_target": relationship_target,
                                                "relationship_visual_distractor": rel_visual_distractor,
                                                "counterexample_relationship_target": counterex_rel_target,
                                                "counterexample_relationship_visual_distractor": counterex_rel_visual_distractor,
                                                "rel_label": rel_label,
                                            }
                                            samples.append(sample)
    return samples


def get_index_key(sample):
    # use alphabetical order of images because we also want to catch cases of flipped example/counterexample:
    images_key = min(sample["img_example"], sample["img_counterexample"]) + "_" + max(sample["img_example"], sample["img_counterexample"])
    if sample["relationship_target"][REL] in RELATIONSHIPS_SPATIAL and sample["rel_label"] == REL:
        return images_key + "_" + sample["relationship_target"][
            SUBJECT] + "_" + sample["relationship_target"][REL] + "_" + sample["relationship_target"][OBJECT]
    else:
        return images_key + "_" + sample["relationship_target"][SUBJECT] + "_" + sample["relationship_target"][sample["rel_label"]]


def generate_eval_sets_from_subject_tuples(
    subject_tuples, split, max_samples, file_name, check_sharpness
):
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

        target_subject, distractor_subject = target_tuple

        # Compute matching images
        is_target = F(SUBJECT).is_in(SYNONYMS[target_subject])
        is_distractor = F(SUBJECT).is_in(SYNONYMS[distractor_subject])
        is_big_enough = F(BOUNDING_BOX)[2] * F(BOUNDING_BOX)[3] > THRESHOLD_MIN_BB_SIZE
        matching_images = dataset.match(
            (F(IMAGE_RELATIONSHIPS).filter(is_target & is_big_enough).length() > 0)
            & (
                F(IMAGE_RELATIONSHIPS).filter(is_distractor & is_big_enough).length()
                > 0
            )
        )

        process_args = []
        matching_images = [sample_to_dict(s) for s in matching_images]
        for example in tqdm(matching_images):
            process_args.append((example,
                    matching_images,
                    target_subject,
                    distractor_subject,
                    check_sharpness))

        with Pool(processes=8) as pool:
            results = pool.starmap(process_sample_subj, tqdm(process_args, total=len(process_args)))

        print("Dropping duplicates...")
        # build index for faster dropping of duplicates
        all_results = []
        for res_process in results:
            for sample in res_process:
                all_results.append((get_index_key(sample), sample))

        eval_set = {}
        for key, sample in tqdm(all_results):
            # Replace current sample if new one has bigger objects
            if key in eval_set.keys():
                duplicate_sample = eval_set[key]
                if get_sum_of_bounding_box_sizes(
                    sample
                ) > get_sum_of_bounding_box_sizes(
                    duplicate_sample
                ):
                    eval_set[key] = sample
            else:
                eval_set[key] = sample
                # show_image_pair(example.filepath, counterexample.filepath, [relationship_target, rel_visual_distractor], [counterex_rel_target, counterex_rel_visual_distractor])

        eval_set = list(eval_set.values())
        if len(eval_set) > 0:
            eval_sets[target_tuple] = eval_set
            print("saving intermediate results..")
            pickle.dump(eval_sets, open(file_name, "wb"))
            print(
                f"\nFound {len(eval_sets[target_tuple])} examples for {target_tuple}.\n"
            )

    return eval_sets


def process_sample_rel_or_obj(
    example,
    rel_label,
    matching_images,
    target_attribute,
    distractor_attribute,
    counterexample_cache,
    eval_set,
    check_sharpness,
    thread_lock_eval_set,
    thread_lock_counterexample_cache,
):
    candidate_relationships = [
        rel
        for rel in example.relationships.detections
        if rel[rel_label] in SYNONYMS[target_attribute]
        and get_bounding_box_size(rel) > THRESHOLD_MIN_BB_SIZE
        and not is_subj_rel_in_image(
            example, rel[SUBJECT], distractor_attribute, rel_label
        )  # check that distractor IS NOT in same image
    ]
    candidate_relationships = drop_synonyms(candidate_relationships, SUBJECT)

    for relationship_target in candidate_relationships:
        target_subject = relationship_target[SUBJECT]

        # check that visual distractor IS in image
        rels_visual_distractor = find_other_subj_with_attr(
            example, relationship_target, distractor_attribute, rel_label
        )
        rels_visual_distractor = drop_synonyms(rels_visual_distractor, SUBJECT)

        if len(rels_visual_distractor) > 0:

            if not check_sharpness or relationships_are_sharp(
                example,
                [relationship_target],
            ):

                # Start looking for counterexample image..
                counterexample_key = get_counterexample_key(
                    relationship_target, SUBJECT
                )

                if counterexample_key not in counterexample_cache:
                    counterx_images = get_counterexample_images_attr(
                        matching_images,
                        distractor_attribute,
                        relationship_target,
                        rel_label,
                    )
                    thread_lock_counterexample_cache.acquire()
                    counterexample_cache[counterexample_key] = counterx_images
                    thread_lock_counterexample_cache.release()

                matching_images_counterexample = counterexample_cache[
                    counterexample_key
                ]

                for counterexample in matching_images_counterexample:

                    counterexample_relationships = [
                        rel
                        for rel in counterexample.relationships.detections
                        if rel[SUBJECT] in SYNONYMS[target_subject]
                        and rel[rel_label] in SYNONYMS[distractor_attribute]
                        and get_bounding_box_size(rel) > THRESHOLD_MIN_BB_SIZE
                    ]
                    counterexample_relationships = drop_synonyms(
                        counterexample_relationships,
                        SUBJECT,
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
                            counterex_rels_visual_distractor,
                            SUBJECT,
                        )
                        if len(counterex_rels_visual_distractor) > 0:
                            for rel_visual_distractor in rels_visual_distractor:
                                for (
                                    counterex_rel_visual_distractor
                                ) in counterex_rels_visual_distractor:
                                    if not check_sharpness or (
                                        relationships_are_sharp(
                                            counterexample,
                                            [
                                                counterex_rel_target,
                                                counterex_rel_visual_distractor,
                                            ],
                                        )
                                        and relationships_are_sharp(
                                            example,
                                            [rel_visual_distractor],
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

                                        # Replace current sample if new one has bigger objects
                                        if duplicate_sample is not None:
                                            if get_sum_of_bounding_box_sizes(
                                                sample
                                            ) > get_sum_of_bounding_box_sizes(
                                                duplicate_sample
                                            ):
                                                thread_lock_eval_set.acquire()
                                                eval_set.remove(duplicate_sample)
                                                eval_set.append(sample)
                                                thread_lock_eval_set.release()
                                        else:
                                            # Add tuple of example and counter-example
                                            thread_lock_eval_set.acquire()
                                            eval_set.append(sample)
                                            thread_lock_eval_set.release()
                                            # show_image_pair(example.filepath, counterexample.filepath, [relationship_target, rel_visual_distractor], [counterex_rel_target, counterex_rel_visual_distractor])


def generate_eval_sets_from_rel_or_object_tuples(
    tuples, rel_label, split, max_samples, file_name, check_sharpness
):
    eval_sets = {}

    dataset = foz.load_zoo_dataset(
        "open-images-v6",
        label_types=["relationships"],
        max_samples=max_samples,
        split=split,
        dataset_name=f"open-images-v6-{split}-{max_samples}",
    )

    for target_tuple in tuples:
        print("Looking for: ", target_tuple)

        eval_set = []
        counterexample_cache = {}
        target_attribute, distractor_attribute = target_tuple

        # Compute matching images
        is_target = F(rel_label).is_in(SYNONYMS[target_attribute])
        is_distractor = F(rel_label).is_in(SYNONYMS[distractor_attribute])
        is_big_enough = F(BOUNDING_BOX)[2] * F(BOUNDING_BOX)[3] > THRESHOLD_MIN_BB_SIZE
        matching_images = dataset.match(
            (F(IMAGE_RELATIONSHIPS).filter(is_target & is_big_enough).length() > 0)
            & (
                F(IMAGE_RELATIONSHIPS).filter(is_distractor & is_big_enough).length()
                > 0
            )
        )

        thread_lock_eval_set = threading.Lock()
        thread_lock_counterexample_cache = threading.Lock()

        threads = []
        for example in tqdm(matching_images):
            t = threading.Thread(
                target=process_sample_rel_or_obj,
                args=(
                    example,
                    rel_label,
                    matching_images,
                    target_attribute,
                    distractor_attribute,
                    counterexample_cache,
                    eval_set,
                    check_sharpness,
                    thread_lock_eval_set,
                    thread_lock_counterexample_cache,
                ),
            )
            t.start()
            threads.append(t)

        for t in tqdm(threads):
            t.join()

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
        "--max-samples",
        type=int,
        default=None,
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
            OBJECTS_TUPLES,
            OBJECT,
            args.split,
            args.max_samples,
            file_name,
            args.check_sharpness,
        )
        pickle.dump(eval_sets, open(file_name, "wb"))

    elif args.eval_set == "relationship_tuples":
        file_name = f"results/rel-{args.split}-{args.max_samples}.p"
        eval_sets = generate_eval_sets_from_rel_or_object_tuples(
            RELATIONSHIPS_TUPLES,
            REL,
            args.split,
            args.max_samples,
            file_name,
            args.check_sharpness,
        )
        pickle.dump(eval_sets, open(file_name, "wb"))
