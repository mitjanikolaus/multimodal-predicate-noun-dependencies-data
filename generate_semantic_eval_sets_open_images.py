import pickle

import fiftyone.zoo as foz
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image as PIL_Image
from matplotlib.patches import Rectangle
from tqdm import tqdm

THRESHOLD_SAME_BOUNDING_BOX = 0.01

NOUN_SYNONYMS = [{"Table", "Desk"}, {"Table", "Coffee table"}]
ATTRIBUTE_SYNONYMS = []


ATTRIBUTE_TUPLES = [
    ("Wooden", "Plastic"),
    ("Wooden", "Transparent"),
    ("Wooden", "(made of)Leather"),
    ("Wooden", "(made of)Textile"),
    ("Smile", "Cry"),
    ("Tree", "Table"),
    ("Plastic", "(made of)Leather"),
    ("Plastic", "(made of)Textile"),
    ("(made of)Leather", "(made of)Textile"),
    ("Car", "Horse"),
    ("Car", "Motorcycle"),
    ("Car", "Bicycle"),
    ("Motorcycle", "Bicycle"),
    ("Horse", "Bicycle"),
    ("Table", "Chair"),
    ("Table", "Tree"),
    ("Dog", "Cat"),
    ("Stand", "Sit"),
    ("Stand", "Walk"),
    ("Stand", "Run"),
    ("Stand", "Lay"),
    ("Stand", "Jump"),
    ("Sit", "Walk"),
    ("Sit", "Run"),
    ("Sit", "Lay"),
    ("Sit", "Jump"),
    ("Walk", "Run"),
    ("Walk", "Jump"),
    ("Walk", "Lay"),
    ("Run", "Jump"),
    ("Run", "Lay"),
    ("Lay", "Jump"),
    ("Glasses", "Sunglasses"),
]

NOUNS_TUPLES = [
    ("Man", "Boy"),
    ("Man", "Woman"),
    ("Woman", "Girl"),
    ("Table", "Chair"),
    ("Desk", "Chair"),
    ("Bottle", "Coffee cup"),
    ("Bottle", "Flowerpot"),
    ("Coffee cup", "Flowerpot"),
    ("Coffee cup", "Wine glass"),
    ("Piano", "Guitar"),
    ("Backpack", "Suitcase"),
]
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
nouns_names = [name for name, _ in nouns_counter]

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
attributes_names = [name for name, _ in attributes_counter]

for attr1, attr2 in ATTRIBUTE_TUPLES:
    assert attr1 in attributes_names
    assert attr2 in attributes_names

for noun1, noun2 in NOUNS_TUPLES:
    assert noun1 in nouns_names
    assert noun2 in nouns_names


def show_image_pair(
    image_1_path, image_2_path, regions_and_attributes_1, regions_and_attributes_2
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
    ax = plt.gca()
    colors = ["green", "red"]
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
            relationship.Label1 + f" ({relationship.Label2})",
            style="italic",
            bbox={"facecolor": "white", "alpha": 0.7, "pad": 10},
        )

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
            relationship.Label1 + f" ({relationship.Label2})",
            style="italic",
            bbox={"facecolor": "white", "alpha": 0.7, "pad": 10},
        )

    plt.tick_params(labelbottom="off", labelleft="off")
    plt.show()


def is_subj_attr_in_image(sample, subject, attribute):
    if sample.relationships:
        for relationship in sample.relationships.detections:
            if relationship.Label1 == subject and relationship.Label2 == attribute:
                return relationship

    return False


def find_other_subj_with_attr(sample, relationship_target, attribute):
    if sample.relationships:
        for relationship in sample.relationships.detections:
            if (
                relationship.Label1 != relationship_target.Label1
                and relationship.Label2 == attribute
            ):
                # verify that they are not synonyms:
                if (
                    not {relationship_target.Label1, relationship.Label1}
                    in NOUN_SYNONYMS
                ):
                    # verify that the two subjects are not duplicate annotations
                    # (actually the (almost) the same bounding box)!
                    diffs = [
                        abs(b1 - b2)
                        for b1, b2 in zip(
                            relationship.bounding_box, relationship_target.bounding_box
                        )
                    ]
                    if not np.all(
                        [diff < THRESHOLD_SAME_BOUNDING_BOX for diff in diffs]
                    ):
                        return relationship

    return None


def find_subj_with_other_attr(sample, subject, relationship_target):
    if sample.relationships:
        for relationship in sample.relationships.detections:
            if (
                relationship.Label1 == subject
                and relationship.Label2 != relationship_target.Label2
            ):
                # verify that they are not synonyms:
                if (
                    not {relationship_target.Label2, relationship.Label2}
                    in ATTRIBUTE_SYNONYMS
                ):
                    # verify that the two subjects are not duplicate annotations
                    # (actually the (almost) the same bounding box)!
                    diffs = [
                        abs(b1 - b2)
                        for b1, b2 in zip(
                            relationship.bounding_box, relationship_target.bounding_box
                        )
                    ]
                    if not np.all(
                        [diff < THRESHOLD_SAME_BOUNDING_BOX for diff in diffs]
                    ):
                        return relationship

    return None


def generate_eval_set_attribute_noun_dependencies_nouns(dataset, noun_tuples):
    eval_sets = {tuple: [] for tuple in noun_tuples}

    for target_tuple in noun_tuples:
        print("Looking for: ", target_tuple)
        target_noun, distractor_noun = target_tuple
        for sample_target in tqdm(dataset):
            if sample_target.relationships:
                for relationship_target in sample_target.relationships.detections:

                    target_subject = relationship_target.Label1

                    # TODO: check for noun synonyms!
                    if target_noun == target_subject:
                        target_attribute = relationship_target.Label2

                        # check that distractor IS NOT in same image:
                        if not is_subj_attr_in_image(
                            sample_target, distractor_noun, target_attribute
                        ):
                            # check that visual distractor IS in image
                            relationship_visual_distractor = find_subj_with_other_attr(
                                sample_target, distractor_noun, relationship_target
                            )
                            if relationship_visual_distractor:
                                # print("Found match: ", sample_target.image)

                                # print("Looking for counterexample image.. ")
                                for counterexample in dataset:

                                    # check that distractor IS in the distractor image:
                                    counterexample_relationship_target = is_subj_attr_in_image(
                                        counterexample,
                                        distractor_noun,
                                        target_attribute,
                                    )
                                    if counterexample_relationship_target:
                                        # TODO: distractor subject == target subject?!

                                        # check that target IS NOT in same image:
                                        if not is_subj_attr_in_image(
                                            counterexample,
                                            target_subject,
                                            target_attribute,
                                        ):

                                            # check that visual distractor IS in image
                                            counterexample_relationship_visual_distractor = find_subj_with_other_attr(
                                                counterexample,
                                                target_noun,
                                                counterexample_relationship_target,
                                            )
                                            # TODO: enforce that distractor subjects are the same?
                                            if counterexample_relationship_visual_distractor:  # and distractor_visual_distractor_subject.synsets[0].name == visual_distractor_subject.synsets[0].name:
                                                # print(f"Found minimal pair: {sample_target.open_images_id} {sample_distractor.open_images_id}")
                                                # show_image_pair(sample_target.filepath, counterexample.filepath, [relationship_target, relationship_visual_distractor], [counterexample_relationship_target, counterexample_relationship_visual_distractor])

                                                # Add example and counter-example
                                                eval_sets[target_tuple].append(
                                                    {
                                                        "img_example": sample_target.filepath,
                                                        "img_counterexample": counterexample.filepath,
                                                        "relationship_target": relationship_target,
                                                        "relationship_visual_distractor": relationship_visual_distractor,
                                                        "counterexample_relationship_target": counterexample_relationship_target,
                                                        "counterexample_relationship_visual_distractor": counterexample_relationship_visual_distractor,
                                                    }
                                                )

        print(f"Found {len(eval_sets[target_tuple])} examples for {target_tuple}.\n\n")

    return eval_sets


def generate_eval_set_attribute_noun_dependencies(dataset, attribute_tuples):
    eval_sets = {tuple: [] for tuple in attribute_tuples}

    for target_tuple in attribute_tuples:
        print("Looking for: ", target_tuple)
        target_attribute, distractor_attribute = target_tuple
        for sample_target in tqdm(dataset):
            if sample_target.relationships:
                for relationship_target in sample_target.relationships.detections:

                    subject_attribute = relationship_target.Label2

                    # TODO: check for attribute synonyms!
                    if target_attribute == subject_attribute:
                        target_subject = relationship_target.Label1

                        # check that distractor IS NOT in same image:
                        if not is_subj_attr_in_image(
                            sample_target, target_subject, distractor_attribute
                        ):

                            # check that visual distractor IS in image
                            relationship_visual_distractor = find_other_subj_with_attr(
                                sample_target, relationship_target, distractor_attribute
                            )
                            if relationship_visual_distractor:
                                # print("Found match: ", scene_graph_target.image)

                                # print("Looking for counterexample image.. ")
                                for counterexample in dataset:

                                    # check that distractor IS in the distractor image:
                                    counterexample_relationship_target = is_subj_attr_in_image(
                                        counterexample,
                                        target_subject,
                                        distractor_attribute,
                                    )
                                    if counterexample_relationship_target:
                                        # TODO: distractor subject == target subject?!

                                        # check that target IS NOT in same image:
                                        if not is_subj_attr_in_image(
                                            counterexample,
                                            target_subject,
                                            target_attribute,
                                        ):

                                            # check that visual distractor IS in image
                                            counterexample_relationship_visual_distractor = find_other_subj_with_attr(
                                                counterexample,
                                                counterexample_relationship_target,
                                                target_attribute,
                                            )
                                            # TODO: enforce that distractor subjects are the same?
                                            if counterexample_relationship_visual_distractor:  # and distractor_visual_distractor_subject.synsets[0].name == visual_distractor_subject.synsets[0].name:
                                                # print(f"Found minimal pair: {sample_target.open_images_id} {sample_distractor.open_images_id}")
                                                # show_image_pair(sample_target.filepath, counterexample.filepath, [relationship_target, relationship_visual_distractor], [counterexample_relationship_target, counterexample_relationship_visual_distractor])

                                                # Add tuple of example and counter-example
                                                eval_sets[target_tuple].append(
                                                    {
                                                        "img_example": sample_target.filepath,
                                                        "img_counterexample": counterexample.filepath,
                                                        "relationship_target": relationship_target,
                                                        "relationship_visual_distractor": relationship_visual_distractor,
                                                        "counterexample_relationship_target": counterexample_relationship_target,
                                                        "counterexample_relationship_visual_distractor": counterexample_relationship_visual_distractor,
                                                    }
                                                )

        print(f"Found {len(eval_sets[target_tuple])} examples for {target_tuple}.\n\n")

    return eval_sets


if __name__ == "__main__":
    max_samples = 1000
    dataset = foz.load_zoo_dataset(
        "open-images-v6",
        split="test",
        label_types=["relationships"],
        max_samples=max_samples,
    )

    # eval_sets_based_on_nouns = generate_eval_set_attribute_noun_dependencies_nouns(
    #     dataset, NOUNS_TUPLES
    # )
    #
    # pickle.dump(eval_sets_based_on_nouns, open(f"data/noun-{max_samples}.p", "wb"))

    eval_sets = generate_eval_set_attribute_noun_dependencies(dataset, ATTRIBUTE_TUPLES)

    pickle.dump(eval_sets, open(f"data/attribute-{max_samples}.p", "wb"))

    # eval_sets = pickle.load(open("data/attribute_noun.p", "rb"))
    #
    # for tuple, eval_set in eval_sets.items():
    #     if len(eval_set) > 0:
    #         print(f"{tuple} ({len(eval_set)} examples)")
    #         for example, counterexample in eval_set:
    #             show_image_pair(example["img_target"], example["img_distractor"], [example["relationship_target"], example["relationship_visual_distractor"]],
    #                             [counterexample["relationship_target"], counterexample["relationship_visual_distractor"]])
