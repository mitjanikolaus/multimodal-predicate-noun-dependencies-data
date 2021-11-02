import itertools
import os

import fiftyone
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from PIL import Image as PIL_Image, ImageFilter, ImageStat

from time import time

SUBJECT = "Label1"
REL = "label"
OBJECT = "Label2"

IMAGE_RELATIONSHIPS = "relationships.detections"
BOUNDING_BOX = "bounding_box"


def get_tuples_no_duplicates(names):
    all_tuples = [
        (a1, a2) for a1, a2 in list(itertools.product(names, names)) if a1 != a2
    ]
    tuples = []
    for (a1, a2) in all_tuples:
        if not (a2, a1) in tuples:
            tuples.append((a1, a2))
    return tuples


# Objects (Label2)

OBJECTS_TEXTURES = [
    "Wooden",
    "Plastic",
    "Transparent",
    "(made of)Leather",
    "(made of)Textile",
]
OBJECTS_TEXTURES_TUPLES = get_tuples_no_duplicates(OBJECTS_TEXTURES)

OBJECTS_INSTRUMENTS = [
    "French horn",
    "Piano",
    "Saxophone",
    "Guitar",
    "Violin",
    "Trumpet",
    "Accordion",
    "Microphone",
    "Cello",
    "Trombone",
    "Flute",
    "Drum",
    "Musical keyboard",
    "Banjo",
]

OBJECTS_VEHICLES = [
    "Car",
    "Motorcycle",
    "Bicycle",
    "Horse",
    "Roller skates",
    "Skateboard",
    "Cart",
    "Bus",
    "Wheelchair",
    "Boat",
    "Canoe",
    "Truck",
    "Train",
    "Tank",
    "Airplane",
    "Van",
]

OBJECTS_ANIMALS = ["Dog", "Cat", "Horse", "Elephant"]

OBJECTS_VERBS = [
    "Smile",
    "Cry",
    "Talk",
    "Sing",
    "Sit",
    "Walk",
    "Lay",
    "Jump",
    "Run",
    "Stand",
]

OBJECTS_FURNITURE = ["Table", "Chair", "Bench", "Bed", "Sofa bed", "Billiard table"]

OBJECTS_OTHERS = [
    "Glasses",
    "Bottle",
    "Wine glass",
    "Coffee cup",
    "Sun hat",
    "Bicycle helmet",
    "High heels",
    "Necklace",
    "Scarf",
    "Belt",
    "Swim cap",
    "Handbag",
    "Crown",
    "Football",
    "Baseball glove",
    "Baseball bat",
    "Racket",
    "Surfboard",
    "Paddle",
    "Camera",
    "Mobile phone",
    "Houseplant",
    "Coffee",
    "Tea",
    "Cocktail",
    "Juice",
    "Cake",
    "Strawberry",
    "Wine",
    "Beer",
    "Woman",
    "Man",
    "Tent",
    "Tree",
    "Girl",
    "Boy",
    "Balloon",
    "Rifle",
    "Earrings",
    "Teddy bear",
    "Doll",
    "Bicycle wheel",
    "Ski",
    "Backpack",
    "Ice cream",
    "Book",
    "Cutting board",
    "Watch",
    "Tripod",
    "Rose",
]

OBJECTS_OTHERS += (
    OBJECTS_INSTRUMENTS
    + OBJECTS_VEHICLES
    + OBJECTS_ANIMALS
    + OBJECTS_VERBS
    + OBJECTS_FURNITURE
)
OBJECTS_OTHERS_TUPLES = get_tuples_no_duplicates(OBJECTS_OTHERS)

OBJECTS_TUPLES = OBJECTS_OTHERS_TUPLES + OBJECTS_TEXTURES_TUPLES

# Nouns (Label1)
SUBJECTS_FRUITS = [
    "Orange",
    "Strawberry",
    "Lemon",
    "Apple",
    "Coconut",
]

SUBJECTS_ACCESSORIES = ["Handbag", "Backpack", "Suitcase"]

SUBJECTS_FURNITURE = ["Chair", "Table", "Sofa bed", "Bed", "Bench"]

SUBJECTS_INSTRUMENTS = ["Piano", "Guitar", "Drum", "Violin"]

SUBJECTS_ANIMALS = ["Dog", "Cat"]

SUBJECTS_OTHERS = [
    "Wine glass",
    "Cake",
    "Beer",
    "Mug",
    "Bottle",
    "Bowl",
    "Flowerpot",
    "Chopsticks",
    "Platter",
    "Ski",
    "Candle",
    "Fork",
    "Spoon",
]

SUBJECTS = (
    SUBJECTS_OTHERS
    + SUBJECTS_FURNITURE
    + SUBJECTS_FRUITS
    + SUBJECTS_ACCESSORIES
    + SUBJECTS_INSTRUMENTS
    + SUBJECTS_ANIMALS
)

SUBJECTS_GENERAL_TUPLES = get_tuples_no_duplicates(SUBJECTS)

SUBJECTS_OTHERS_TUPLES = [
    ("Man", "Woman"),
    ("Man", "Girl"),
    ("Woman", "Boy"),
    ("Girl", "Boy"),
]

SUBJECT_TUPLES = SUBJECTS_GENERAL_TUPLES + SUBJECTS_OTHERS_TUPLES

# Relationships (.label)
RELATIONSHIPS_SPATIAL = ["at", "contain", "holds", "on", "hang", "inside_of", "under"]
RELATIONSHIPS_SPATIAL_TUPLES = get_tuples_no_duplicates(RELATIONSHIPS_SPATIAL)

RELATIONSHIPS_BALL = ["throw", "catch", "kick", "holds", "hits"]
RELATIONSHIPS_BALL_TUPLES = get_tuples_no_duplicates(RELATIONSHIPS_BALL)

RELATIONSHIPS_OTHERS = [
    "eat",
    "drink",
    "read",
    "dance",
    "kiss",
    "skateboard",
    "surf",
    "ride",
    "hug",
    "plays",
]
RELATIONSHIPS_OTHERS_TUPLES = get_tuples_no_duplicates(RELATIONSHIPS_OTHERS)

RELATIONSHIPS = RELATIONSHIPS_SPATIAL + RELATIONSHIPS_BALL + RELATIONSHIPS_OTHERS
RELATIONSHIPS_TUPLES = (
    RELATIONSHIPS_SPATIAL_TUPLES
    + RELATIONSHIPS_BALL_TUPLES
    + RELATIONSHIPS_OTHERS_TUPLES
)

subjects_counter = pd.read_csv(
    "data/subject_occurrences.csv",
    index_col=None,
    header=None,
    names=["subject", "count"],
)
SUBJECT_NAMES = list(subjects_counter["subject"].values)

relationships_counter = pd.read_csv(
    "data/rel_occurrences.csv", index_col=None, header=None, names=["rel", "count"]
)
RELATIONSHIP_NAMES = list(relationships_counter["rel"].values)

objects_counter = pd.read_csv(
    "data/obj_occurrences.csv", index_col=None, header=None, names=["obj", "count"]
)
OBJECT_NAMES = list(objects_counter["obj"].values)

for obj1, obj2 in OBJECTS_TUPLES:
    assert obj1 in OBJECT_NAMES, f"{obj1} is misspelled"
    assert obj2 in OBJECT_NAMES, f"{obj2} is misspelled"

for subj1, subj2 in SUBJECT_TUPLES:
    assert subj1 in SUBJECT_NAMES, f"{subj1} is misspelled"
    assert subj2 in SUBJECT_NAMES, f"{subj2} is misspelled"

SYNONYMS_LIST = [
    ["Table", "Desk", "Coffee table"],
    ["Mug", "Coffee cup"],
    ["Glasses", "Sunglasses", "Goggles"],
    ["Sun hat", "Fedora", "Cowboy hat", "Sombrero"],
    ["Bicycle helmet", "Football helmet"],
    ["High heels", "Sandal", "Boot"],
    ["Racket", "Tennis racket", "Table tennis racket"],
    ["Crown", "Tiara"],
    ["Handbag", "Briefcase"],
    ["Cart", "Golf cart"],
    ["Football", "Volleyball (Ball)", "Rugby ball", "Cricket ball", "Tennis ball"],
    ["Tree", "Palm tree"],
]

SYNONYMS = {name: [name] for name in SUBJECT_NAMES + OBJECT_NAMES + RELATIONSHIP_NAMES}
for synonyms in SYNONYMS_LIST:
    SYNONYMS.update({item: synonyms for item in synonyms})

VALID_NAMES = {"label": RELATIONSHIPS, "Label2": OBJECT_NAMES}

REFERENCE_TIME = time()
TIME_LOG_THRESHOLD = 0.1


def log_time(message, reference_time=None):
    if reference_time:
        ref = reference_time
    else:
        global REFERENCE_TIME
        ref = REFERENCE_TIME

    current_time = time()
    timedelta = current_time - ref

    if timedelta > TIME_LOG_THRESHOLD:
        line = "=" * 40
        print(line)
        print(f"{message} | Time passed: {timedelta:.1f}s")
        print(line)
        print()

    if not reference_time:
        REFERENCE_TIME = current_time


def get_local_image_path(img_path):
    return os.path.join(fiftyone.config.dataset_zoo_dir, img_path.split("fiftyone/")[1])


# Threshold for overlap of 2 bounding boxes
THRESHOLD_SAME_BOUNDING_BOX = 0.02


def high_bounding_box_overlap(bb1, bb2, threshold=THRESHOLD_SAME_BOUNDING_BOX):
    """verify that bounding boxes are not duplicate annotations (actually the (almost) the same bounding box)"""
    diffs = [abs(b1 - b2) for b1, b2 in zip(bb1, bb2)]
    if np.all([diff < threshold for diff in diffs]):
        return True
    return False


def show_image(
    image_1_path,
    regions_and_attributes_1=None,
):
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    img_1_data = PIL_Image.open(image_1_path)

    plt.imshow(img_1_data)

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
            ax.text(
                bb[0] * img_1_data.width,
                bb[1] * img_1_data.height,
                f"{relationship[SUBJECT]} {relationship[REL]} {relationship[OBJECT]} ",
                bbox={"facecolor": "white", "alpha": 0.7, "pad": 10},
            )

    plt.tick_params(labelbottom="off", labelleft="off")
    plt.show()


def show_image_pair(
    image_1_path,
    image_2_path,
    regions_and_attributes_1=None,
    regions_and_attributes_2=None,
    target_sentence=None,
    distractor_sentence=None,
    result_example=None,
    result_counterexample=None,
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
            ax.text(
                bb[0] * img_1_data.width,
                bb[1] * img_1_data.height,
                f"{relationship[SUBJECT]} {relationship[REL]} {relationship[OBJECT]} ",
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
            ax.text(
                bb[0] * img_2_data.width + x_offset,
                bb[1] * img_2_data.height,
                f"{relationship[SUBJECT]} {relationship[REL]} {relationship[OBJECT]} ",
                bbox={"facecolor": "white", "alpha": 0.7, "pad": 10},
            )

    plt.tick_params(labelbottom="off", labelleft="off")

    ax.axis("off")
    ax.text(
        0.5,
        -0.12,
        f"Target sentence: {target_sentence}\nDistractor sentence: {distractor_sentence}\n"
        f"Result: {result_example}\nResult (counterexample): {result_counterexample}",
        size=12,
        ha="center",
        transform=ax.transAxes,
    )

    plt.show()


def show_sample(
    sample,
    target_sentence=None,
    distractor_sentence=None,
    result_example=None,
    result_counterexample=None,
):
    img_1_path = get_local_image_path(sample["img_example"])
    img_2_path = get_local_image_path(sample["img_counterexample"])

    show_image_pair(
        img_1_path,
        img_2_path,
        [sample["relationship_target"], sample["relationship_visual_distractor"]],
        [
            sample["counterexample_relationship_target"],
            sample["counterexample_relationship_visual_distractor"],
        ],
        target_sentence,
        distractor_sentence,
        result_example,
        result_counterexample,
    )


def generate_sentence_from_triplet(subject, predicate, object):
    if object in OBJECTS_VERBS:
        if object.endswith("t"):
            object += "ting"
        elif object.endswith("e"):
            object = object[:-1] + "ing"
        else:
            object += "ing"
    if object in OBJECTS_ANIMALS + OBJECTS_INSTRUMENTS + OBJECTS_VEHICLES + OBJECTS_OTHERS:
        if object not in ["Glasses"]:
            object = "a " + object

    if "(made of)" in object:
        object = object.replace("(made of)", " made of ")

    sentence = f"a {subject} {predicate} {object}"

    # Add full stop
    sentence = sentence + "."

    # Lower case
    sentence = sentence.lower()

    return sentence


def get_target_and_distractor_sentence(sample):
    # TODO: consider rel_label?
    text_target = generate_sentence_from_triplet(
        SYNONYMS[sample["relationship_target"][SUBJECT]][0],
        sample["relationship_target"].label,
        SYNONYMS[sample["relationship_target"][OBJECT]][0],
    )
    text_distractor = generate_sentence_from_triplet(
        SYNONYMS[sample["counterexample_relationship_target"][SUBJECT]][0],
        sample["counterexample_relationship_target"].label,
        SYNONYMS[sample["counterexample_relationship_target"][OBJECT]][0],
    )

    return text_target, text_distractor
