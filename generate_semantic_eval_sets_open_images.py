import argparse
import pickle

from fiftyone import ViewField as F
import fiftyone.zoo as foz
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image as PIL_Image, ImageFilter, ImageStat
from matplotlib.patches import Rectangle
from tqdm import tqdm

from utils import get_tuples_no_duplicates

THRESHOLD_SAME_BOUNDING_BOX = 0.02

# Bounding boxes of objects should be at least 10% of image in width and height
THRESHOLD_MIN_BOUNDING_BOX_WIDTH = 0.1
THRESHOLD_MIN_BOUNDING_BOX_HEIGHT = 0.1

# Bounding box sharpness quotient (relative to whole image)
THRESHOLD_MIN_BOUNDING_BOX_SHARPNESS = 0.6

####ATTRIBUTES (Label2)
OBJECTS = ["Houseplant", "Coffee", "Tea", "Cake"]
OBJECTS_TUPLES = get_tuples_no_duplicates(OBJECTS)

TEXTURES = ["Wooden", "Plastic", "Transparent", "(made of)Leather", "(made of)Textile"]
TEXTURES_TUPLES = get_tuples_no_duplicates(TEXTURES)

INSTRUMENTS = [
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
    "Harp",
    "Flute",
    "Drum",
    "Musical keyboard",
]

VEHICLES = [
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
]

ANIMALS = ["Dog", "Cat", "Horse", "Elephant"]

VERBS = ["Smile", "Cry", "Talk", "Sing", "Sit", "Walk", "Lay", "Jump", "Run", "Stand"]

BALLS = ["Football", "Volleyball (Ball)", "Rugby ball", "Cricket ball", "Tennis ball"]

FURNITURE = ["Table", "Chair", "Wheelchair"]

ATTRIBUTES_PERSONS = [
    "Glasses",
    "Sun hat",
    "Bicycle helmet",
    "High heels",
    "Necklace",
    "Scarf",
    "Swim cap",
    "Handbag",
    "Crown",
    "Roller skates",
    "Skateboard",
    "Baseball glove",
    "Baseball bat",
    "Racket",
    "Surfboard",
    "Paddle",
    "Dog",
    "Table",
    "Chair",
    "Camera",
]
ATTRIBUTES_PERSONS += INSTRUMENTS + VEHICLES + BALLS + ANIMALS + VERBS + FURNITURE
ATTRIBUTES_PERSONS_TUPLES = get_tuples_no_duplicates(ATTRIBUTES_PERSONS)

ATTRIBUTE_TUPLES = ATTRIBUTES_PERSONS_TUPLES + TEXTURES_TUPLES + OBJECTS_TUPLES

### Nouns (Label1)
NOUNS_FRUITS = [
    "Orange",
    "Grapefruit",
    "Strawberry",
    "Lemon",
    "Grape",
    "Peach",
    "Apple",
    "Pear",
    "Coconut",
]

NOUNS_ACCESSORIES = ["Handbag", "Backpack", "Suitcase"]

NOUNS_FURNITURE = ["Chair", "Table", "Sofa bed", "Bed"]

NOUNS_INSTRUMENTS = ["Piano", "Guitar", "Drum", "Violin"]

NOUNS_ANIMALS = ["Dog"]

NOUNS_OBJECTS = [
    "Wine glass",
    "Mug",
    "Bottle",
    "Bowl",
    "Flowerpot",
    "Chopsticks",
    "Platter",
    "Ski",
]

NOUNS_OBJECTS += (
    NOUNS_FURNITURE
    + NOUNS_FRUITS
    + NOUNS_ACCESSORIES
    + NOUNS_INSTRUMENTS
    + NOUNS_ANIMALS
)

NOUNS_OBJECTS_TUPLES = get_tuples_no_duplicates(NOUNS_OBJECTS)

NOUNS_TUPLES_OTHER = [
    ("Man", "Woman"),
]

NOUN_TUPLES = NOUNS_OBJECTS_TUPLES + NOUNS_TUPLES_OTHER

nouns_counter = [
    ("Man", 31890),
    ("Woman", 19579),
    ("Girl", 15104),
    ("Boy", 4375),
    ("Chair", 2665),
    ("Table", 2526),
    ("Bottle", 1139),
    ("Coffee cup", 677),
    ("Desk", 472),
    ("Flowerpot", 455),
    ("Piano", 323),
    ("Wine glass", 319),
    ("Guitar", 316),
    ("Handbag", 287),
    ("Orange", 244),
    ("Sofa bed", 207),
    ("Grapefruit", 175),
    ("Mug", 164),
    ("Bed", 164),
    ("Drum", 142),
    ("Backpack", 136),
    ("Suitcase", 134),
    ("Strawberry", 124),
    ("Coffee table", 121),
    ("Platter", 117),
    ("Violin", 96),
    ("Lemon", 86),
    ("Ski", 76),
    ("Grape", 62),
    ("Chopsticks", 54),
    ("Peach", 52),
    ("Apple", 50),
    ("Pear", 49),
    ("Dog", 42),
    ("Coconut", 40),
    ("Bowl", 37),
    ("Common fig", 34),
    ("Briefcase", 30),
    ("Fork", 28),
    ("Spoon", 25),
    ("Cat", 24),
    ("Knife", 16),
    ("Bench", 15),
    ("Beer", 12),
    ("Pomegranate", 10),
    ("Mango", 9),
    ("Tomato", 7),
    ("Kitchen knife", 6),
    ("Candle", 6),
    ("Cake", 5),
    ("Picnic basket", 5),
    ("Artichoke", 4),
    ("Teapot", 3),
    ("Pineapple", 2),
    ("Carrot", 2),
    ("Cheese", 1),
    ("Pizza", 1),
    ("Cucumber", 1),
    ("Bread", 1),
]
NOUN_NAMES = [name for name, _ in nouns_counter]

relationships_counter = [
    ("is", 68343),
    ("wears", 4668),
    ("at", 2095),
    ("contain", 1618),
    ("holds", 1587),
    ("ride", 1071),
    ("on", 985),
    ("hang", 835),
    ("plays", 460),
    ("interacts_with", 339),
    ("inside_of", 281),
    ("skateboard", 119),
    ("surf", 86),
    ("hits", 66),
    ("kick", 54),
    ("throw", 49),
    ("catch", 25),
    ("drink", 19),
    ("read", 15),
    ("eat", 14),
    ("under", 10),
    ("ski", 7),
]

attributes_counter = [
    ("Stand", 29768),
    ("Smile", 12492),
    ("Sit", 11017),
    ("Wooden", 4186),
    ("Walk", 2723),
    ("Table", 1850),
    ("Run", 1724),
    ("Plastic", 1684),
    ("Glasses", 931),
    ("Talk", 922),
    ("Transparent", 917),
    ("Lay", 822),
    ("Tree", 807),
    ("(made of)Leather", 707),
    ("Jump", 608),
    ("Sunglasses", 544),
    ("Roller skates", 489),
    ("(made of)Textile", 481),
    ("Houseplant", 432),
    ("Horse", 414),
    ("Coffee", 390),
    ("Car", 376),
    ("High heels", 315),
    ("Sandal", 314),
    ("Bicycle", 305),
    ("Tea", 289),
    ("Goggles", 288),
    ("Desk", 278),
    ("Sun hat", 264),
    ("Sing", 237),
    ("Fedora", 228),
    ("Dog", 222),
    ("Wine", 210),
    ("Boat", 209),
    ("Bicycle helmet", 183),
    ("Cowboy hat", 180),
    ("Canoe", 158),
    ("Football helmet", 157),
    ("Wheelchair", 152),
    ("Chair", 151),
    ("Boot", 149),
    ("Guitar", 144),
    ("Skateboard", 142),
    ("Football", 136),
    ("Necklace", 120),
    ("Baseball glove", 114),
    ("Surfboard", 110),
    ("Cake", 108),
    ("Tennis racket", 105),
    ("Paddle", 103),
    ("Motorcycle", 93),
    ("Scarf", 89),
    ("Trumpet", 87),
    ("Table tennis racket", 86),
    ("Swim cap", 82),
    ("Cart", 81),
    ("Racket", 81),
    ("French horn", 80),
    ("Violin", 80),
    ("Camera", 69),
    ("Coffee table", 66),
    ("Bed", 62),
    ("Beer", 61),
    ("Bus", 60),
    ("Volleyball (Ball)", 56),
    ("Cry", 55),
    ("Sushi", 55),
    ("Handbag", 53),
    ("Tiara", 53),
    ("Microphone", 52),
    ("Cello", 52),
    ("Trombone", 51),
    ("Baseball bat", 51),
    ("Crown", 48),
    ("Cocktail", 48),
    ("Juice", 46),
    ("Accordion", 43),
    ("Segway", 43),
    ("Saxophone", 43),
    ("Balance beam", 43),
    ("Billiard table", 42),
    ("Sombrero", 41),
    ("Piano", 41),
    ("Rifle", 40),
    ("Harp", 38),
    ("Flute", 38),
    ("Drum", 35),
    ("Musical keyboard", 33),
    ("Bicycle wheel", 32),
    ("Bottle", 31),
    ("Hiking equipment", 30),
    ("Earrings", 29),
    ("Rugby ball", 28),
    ("Bow and arrow", 25),
    ("Palm tree", 24),
    ("Rose", 24),
    ("Wine glass", 23),
    ("Golf cart", 23),
    ("Elephant", 22),
    ("Balloon", 22),
    ("Belt", 21),
    ("Oyster", 20),
    ("Infant bed", 19),
    ("Watermelon", 19),
    ("Ski", 17),
    ("Cutting board", 17),
    ("Book", 15),
    ("Salad", 15),
    ("Tripod", 15),
    ("Orange", 15),
    ("Shotgun", 14),
    ("Dumbbell", 14),
    ("Grape", 13),
    ("Strawberry", 13),
    ("Dog bed", 12),
    ("Coffee cup", 11),
    ("Harpsichord", 11),
    ("Mobile phone", 11),
    ("Gondola", 10),
    ("Cat", 10),
    ("Apple", 10),
    ("Truck", 10),
    ("Grapefruit", 10),
    ("Bowling equipment", 9),
    ("Horizontal bar", 9),
    ("Tennis ball", 9),
    ("Stationary bicycle", 9),
    ("Tent", 8),
    ("Airplane", 8),
    ("Watch", 8),
    ("Organ (Musical Instrument)", 8),
    ("Ladder", 8),
    ("Sofa bed", 7),
    ("Muffin", 7),
    ("Suitcase", 7),
    ("Unicycle", 6),
    ("Backpack", 6),
    ("Binoculars", 6),
    ("Van", 6),
    ("Doll", 6),
    ("Handgun", 5),
    ("Limousine", 5),
    ("Cake stand", 5),
    ("Lobster", 5),
    ("Ambulance", 5),
    ("Panda", 5),
    ("Cricket ball", 5),
    ("Countertop", 5),
    ("Stool", 4),
    ("Banjo", 4),
    ("Cat furniture", 4),
    ("Indoor rower", 4),
    ("Monkey", 4),
    ("Bench", 4),
    ("Cannon", 3),
    ("Helicopter", 3),
    ("Oboe", 3),
    ("Sword", 3),
    ("Tank", 3),
    ("Flying disc", 3),
    ("Teddy bear", 3),
    ("Washing machine", 3),
    ("Treadmill", 2),
    ("Box", 2),
    ("Ice cream", 2),
    ("Taxi", 2),
    ("Studio couch", 2),
    ("Train", 2),
    ("Carrot", 2),
    ("Candy", 2),
    ("Wok", 2),
    ("Common sunflower", 2),
    ("Sea lion", 2),
    ("Stethoscope", 2),
    ("Jet ski", 2),
    ("Pen", 2),
    ("Kitchen & dining room table", 2),
    ("Pomegranate", 2),
    ("Milk", 1),
    ("Plate", 1),
    ("Honeycomb", 1),
    ("Egg (Food)", 1),
    ("Lizard", 1),
    ("Loveseat", 1),
    ("Dolphin", 1),
    ("Whale", 1),
    ("Brown bear", 1),
    ("Picnic basket", 1),
    ("Plastic bag", 1),
    ("Punching bag", 1),
    ("Lemon", 1),
    ("Cheese", 1),
    ("Cupboard", 1),
    ("Personal flotation device", 1),
    ("Snowmobile", 1),
    ("Flowerpot", 1),
    ("Broccoli", 1),
    ("Cucumber", 1),
    ("Christmas tree", 1),
    ("Hamster", 1),
    ("Pasta", 1),
    ("Shark", 1),
    ("Kite", 1),
    ("Tart", 1),
    ("Pumpkin", 1),
    ("Crab", 1),
    ("Mug", 1),
    ("Dinosaur", 1),
    ("Tablet computer", 1),
    ("Bowl", 1),
]
ATTRIBUTES_NAMES = [name for name, _ in attributes_counter]

for attr1, attr2 in ATTRIBUTE_TUPLES:
    assert attr1 in ATTRIBUTES_NAMES, f"{attr1} is misspelled"
    assert attr2 in ATTRIBUTES_NAMES, f"{attr2} is misspelled"

for noun1, noun2 in NOUN_TUPLES:
    assert noun1 in NOUN_NAMES, f"{noun1} is misspelled"
    assert noun2 in NOUN_NAMES, f"{noun2} is misspelled"


SYNONYMS_LIST = [
    ["Man", "Boy"],
    ["Woman", "Girl"],
    ["Table", "Desk", "Coffee table"],
    ["Mug", "Coffee cup"],
    ["Glasses", "Sunglasses", "Goggles"],
    ["Sun hat", "Fedora", "Cowboy hat", "Sombrero"],
    ["Bicycle helmet", "Football helmet"],
    ["High heels", "Sandal", "Boot"],
    ["Racket", "Tennis racket", "Table tennis racket"],
    ["Crown", "Tiara"],
]
SYNONYMS = {name: [name] for name in NOUN_NAMES + ATTRIBUTES_NAMES}
for synonyms in SYNONYMS_LIST:
    SYNONYMS.update({item: synonyms for item in synonyms})


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
                # replace existing rel if new one is bigger
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


def relationship_is_sharp(sample, relationship):
    img_data = PIL_Image.open(sample.filepath)
    sharpness_quotient = get_sharpness_quotient_of_bounding_box(
        img_data, relationship.bounding_box
    )
    return sharpness_quotient > THRESHOLD_MIN_BOUNDING_BOX_SHARPNESS


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

    # img_1_sharpness = get_image_sharpness(img_1_data)
    # img_2_sharpness = get_image_sharpness(img_2_data)
    # print("Image 1 sharpness: ", img_1_sharpness)
    # print("Image 2 sharpness: ", img_2_sharpness)

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
                relationship.Label1
                + f" ({relationship.Label2}) (Sharpness: {sharpness_quotient:.2f}",
                style="italic",
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
                relationship.Label1
                + f" ({relationship.Label2}) (Sharpness: {sharpness_quotient:.2f}",
                style="italic",
                bbox={"facecolor": "white", "alpha": 0.7, "pad": 10},
            )

    plt.tick_params(labelbottom="off", labelleft="off")
    plt.show()


def is_subj_attr_in_image(sample, subject, attribute):
    if sample.relationships:
        for relationship in sample.relationships.detections:
            if (
                relationship.Label1 in SYNONYMS[subject]
                and relationship.Label2 in SYNONYMS[attribute]
            ):
                return relationship

    return False


def find_other_subj_with_attr(sample, relationship_target, attribute):
    relationships = []
    if sample.relationships:
        for relationship in sample.relationships.detections:
            if (
                relationship.Label1 not in SYNONYMS[relationship_target.Label1]
                and relationship.Label2 in SYNONYMS[attribute]
            ):
                # verify that the two subjects are not duplicate annotations
                # (actually the (almost) the same bounding box)!
                diffs = [
                    abs(b1 - b2)
                    for b1, b2 in zip(
                        relationship.bounding_box, relationship_target.bounding_box
                    )
                ]
                if not np.all([diff < THRESHOLD_SAME_BOUNDING_BOX for diff in diffs]):
                    if relationship_is_sharp(sample, relationship):
                        relationships.append(relationship)

    return relationships


def find_subj_with_other_attr(sample, subject, relationship_target):
    relationships = []
    if sample.relationships:
        for relationship in sample.relationships.detections:
            if (
                relationship.Label1 in SYNONYMS[subject]
                and relationship.Label2 not in SYNONYMS[relationship_target.Label2]
            ):
                # verify that the two subjects are not duplicate annotations
                # (actually the (almost) the same bounding box)!
                diffs = [
                    abs(b1 - b2)
                    for b1, b2 in zip(
                        relationship.bounding_box, relationship_target.bounding_box
                    )
                ]
                if not np.all([diff < THRESHOLD_SAME_BOUNDING_BOX for diff in diffs]):
                    if relationship_is_sharp(sample, relationship):
                        relationships.append(relationship)

    return relationships


def get_duplicate_sample(sample, eval_set):
    for existing_sample in eval_set:
        if (
            existing_sample["img_example"] == sample["img_example"]
            and existing_sample["img_counterexample"] == sample["img_counterexample"]
        ) or (
            existing_sample["img_example"] == sample["img_counterexample"]
            and existing_sample["img_counterexample"] == sample["img_example"]
        ):
            if (
                existing_sample["relationship_target"].Label1
                == sample["relationship_target"].Label1
                and existing_sample["relationship_target"].Label2
                == sample["relationship_target"].Label2
            ):
                return existing_sample
    return None


def generate_eval_sets_from_noun_tuples(noun_tuples, max_samples):
    eval_sets = {}

    dataset = foz.load_zoo_dataset(
        "open-images-v6",
        split="test",
        label_types=["relationships"],
        max_samples=max_samples,
    )

    for target_tuple in noun_tuples:
        print("Looking for: ", target_tuple)
        eval_set = []
        target_noun, distractor_noun = target_tuple

        # Compute matching images
        is_target = F("Label1").is_in(SYNONYMS[target_noun])
        is_distractor = F("Label1").is_in(SYNONYMS[distractor_noun])
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
            possible_relationships = [
                rel
                for rel in example.relationships.detections
                if rel.Label1 in SYNONYMS[target_noun]
                and relationship_is_sharp(example, rel)
                and not is_subj_attr_in_image(
                    example, distractor_noun, rel.Label2
                )  # check that distractor IS NOT in same image
            ]
            possible_relationships = drop_synonyms(possible_relationships, "Label2")

            for relationship_target in possible_relationships:
                target_attribute = relationship_target.Label2

                # check that visual distractor IS in image
                rels_visual_distractor = find_subj_with_other_attr(
                    example, distractor_noun, relationship_target
                )

                for rel_visual_distractor in rels_visual_distractor:

                    # Start looking for counterexample image..
                    is_counterexample_relation = F("Label1").is_in(
                        SYNONYMS[distractor_noun]
                    ) & F("Label2").is_in(SYNONYMS[target_attribute])
                    matching_images_counterexample = matching_images.match(
                        F("relationships.detections")
                        .filter(is_counterexample_relation & is_big_enough)
                        .length()
                        > 0
                    )
                    # check that target IS NOT in same image:
                    matching_images_counterexample = [
                        im
                        for im in matching_images_counterexample
                        if not is_subj_attr_in_image(im, target_noun, target_attribute)
                    ]

                    for counterexample in matching_images_counterexample:
                        counterexample_possible_relationships = [
                            rel
                            for rel in counterexample.relationships.detections
                            if rel.Label1 in SYNONYMS[distractor_noun]
                            and rel.Label2 in SYNONYMS[target_attribute]
                            and relationship_is_sharp(counterexample, rel)
                        ]
                        counterexample_possible_relationships = drop_synonyms(
                            counterexample_possible_relationships, "Label2",
                        )

                        for (
                            counterex_relationship_target
                        ) in counterexample_possible_relationships:

                            # check that visual distractor IS in image
                            counterexample_rels_visual_distractor = find_subj_with_other_attr(
                                counterexample,
                                target_noun,
                                counterex_relationship_target,
                            )
                            for (
                                counterex_rel_visual_distractor
                            ) in counterexample_rels_visual_distractor:
                                sample = {
                                    "img_example": example.filepath,
                                    "img_counterexample": counterexample.filepath,
                                    "relationship_target": relationship_target,
                                    "relationship_visual_distractor": rel_visual_distractor,
                                    "counterexample_relationship_target": counterex_relationship_target,
                                    "counterexample_relationship_visual_distractor": counterex_rel_visual_distractor,
                                }
                                duplicate_sample = get_duplicate_sample(
                                    sample, eval_set
                                )

                                # Replace current sample if new one has bigger objects
                                if duplicate_sample is not None:
                                    if get_sum_of_bounding_box_sizes(
                                        sample
                                    ) > get_sum_of_bounding_box_sizes(duplicate_sample):
                                        eval_set.remove(duplicate_sample)
                                        eval_set.append(sample)

                                else:
                                    # print(f"Found minimal pair: {sample_target.open_images_id} {sample_distractor.open_images_id}")
                                    # show_image_pair(example.filepath, counterexample.filepath, [relationship_target, relationship_visual_distractor], [counterexample_relationship_target, counterexample_relationship_visual_distractor])

                                    # Add example and counter-example
                                    eval_set.append(sample)
        if len(eval_set) > 0:
            eval_sets[target_tuple] = eval_set
            print("saving intermediate results..")
            pickle.dump(eval_sets, open(f"results/noun-{max_samples}.p", "wb"))
            print(
                f"\nFound {len(eval_sets[target_tuple])} examples for {target_tuple}.\n"
            )

    return eval_sets


def generate_eval_sets_from_attribute_tuples(attribute_tuples, max_samples):
    dataset = foz.load_zoo_dataset(
        "open-images-v6",
        split="test",
        label_types=["relationships"],
        max_samples=max_samples,
    )

    eval_sets = {}

    for target_tuple in attribute_tuples:
        print("Looking for: ", target_tuple)
        eval_set = []
        target_attribute, distractor_attribute = target_tuple

        # Compute matching images
        is_target = F("Label2").is_in(SYNONYMS[target_attribute])
        is_distractor = F("Label2").is_in(SYNONYMS[distractor_attribute])
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
                if rel.Label2 in SYNONYMS[target_attribute]
                and relationship_is_sharp(example, rel)
                and not is_subj_attr_in_image(
                    example, rel.Label1, distractor_attribute
                )  # check that distractor IS NOT in same image
            ]
            # TODO: necessary?
            relationships = drop_synonyms(relationships, "Label1")

            for relationship_target in relationships:
                target_noun = relationship_target.Label1

                # check that visual distractor IS in image
                rels_visual_distractor = find_other_subj_with_attr(
                    example, relationship_target, distractor_attribute
                )

                for rel_visual_distractor in rels_visual_distractor:
                    # Start looking for counterexample image..
                    is_counterex_rel = F("Label1").is_in(SYNONYMS[target_noun]) & F(
                        "Label2"
                    ).is_in(SYNONYMS[distractor_attribute])
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
                        if not is_subj_attr_in_image(im, target_noun, target_attribute)
                    ]

                    for counterexample in matching_images_counterexample:

                        counterexample_relationships = [
                            rel
                            for rel in counterexample.relationships.detections
                            if rel.Label1 in SYNONYMS[target_noun]
                            and rel.Label2 in SYNONYMS[distractor_attribute]
                            and relationship_is_sharp(counterexample, rel)
                        ]
                        # TODO: necessary?
                        counterexample_relationships = drop_synonyms(
                            counterexample_relationships, "Label1",
                        )

                        for counterexample_rel_target in counterexample_relationships:
                            # check that visual distractor IS in image
                            counterex_rels_visual_distractor = find_other_subj_with_attr(
                                counterexample,
                                counterexample_rel_target,
                                target_attribute,
                            )

                            for (
                                counterex_rel_visual_distractor
                            ) in counterex_rels_visual_distractor:
                                sample = {
                                    "img_example": example.filepath,
                                    "img_counterexample": counterexample.filepath,
                                    "relationship_target": relationship_target,
                                    "relationship_visual_distractor": rel_visual_distractor,
                                    "counterexample_relationship_target": counterexample_rel_target,
                                    "counterexample_relationship_visual_distractor": counterex_rel_visual_distractor,
                                }
                                duplicate_sample = get_duplicate_sample(
                                    sample, eval_set
                                )

                                # Replace current sample if new one has bigger objects
                                if duplicate_sample is not None:
                                    if get_sum_of_bounding_box_sizes(
                                            sample
                                    ) > get_sum_of_bounding_box_sizes(duplicate_sample):
                                        eval_set.remove(duplicate_sample)
                                        eval_set.append(sample)
                                else:
                                    # print(f"Found minimal pair: {sample_target.open_images_id} {sample_distractor.open_images_id}")
                                    # show_image_pair(example.filepath, counterexample.filepath, [relationship_target, relationship_visual_distractor], [counterexample_relationship_target, counterexample_relationship_visual_distractor])

                                    # Add tuple of example and counter-example
                                    eval_set.append(sample)

        if len(eval_set) > 0:
            eval_sets[target_tuple] = eval_set
            print("saving intermediate results..")
            pickle.dump(eval_sets, open(f"results/attribute-{max_samples}.p", "wb"))
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
        choices=["noun_tuples", "attribute_tuples"],
    )
    argparser.add_argument(
        "--max-samples", type=int, default=None,
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.eval_set == "noun_tuples":
        eval_sets_based_on_nouns = generate_eval_sets_from_noun_tuples(
            NOUN_TUPLES, args.max_samples
        )
        pickle.dump(
            eval_sets_based_on_nouns, open(f"results/noun-{args.max_samples}.p", "wb")
        )

    elif args.eval_set == "attribute_tuples":
        eval_sets = generate_eval_sets_from_attribute_tuples(
            ATTRIBUTE_TUPLES, args.max_samples
        )
        pickle.dump(eval_sets, open(f"results/attribute-{args.max_samples}.p", "wb"))
