import itertools
import os
import pickle
from collections import Counter

import fiftyone
import fiftyone.zoo as foz
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image as PIL_Image
from matplotlib.patches import Rectangle
import io
import requests

NOUN_SYNONYMS = [{"Table", "Desk"}, {"Table", "Coffee table"}]

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
    ("Stand", "Sit"),
    ("Stand", "Walk"),
    ("Stand", "Run"),
    ("Stand", "Lay"),
    ("Stand", "Jump"),
]
attributes_counter = [('Stand', 29768), ('Smile', 12492), ('Sit', 11017), ('Wooden', 4186), ('Walk', 2723), ('Table', 1850), ('Run', 1724), ('Plastic', 1684), ('Glasses', 931), ('Talk', 922), ('Transparent', 917), ('Lay', 822), ('Tree', 807), ('(made of)Leather', 707), ('Jump', 608), ('Sunglasses', 544), ('Roller skates', 489), ('(made of)Textile', 481), ('Houseplant', 432), ('Horse', 414), ('Coffee', 390), ('Car', 376), ('High heels', 315), ('Sandal', 314), ('Bicycle', 305), ('Tea', 289), ('Goggles', 288), ('Desk', 278), ('Sun hat', 264), ('Sing', 237), ('Fedora', 228), ('Dog', 222), ('Wine', 210), ('Boat', 209), ('Bicycle helmet', 183), ('Cowboy hat', 180), ('Canoe', 158), ('Football helmet', 157), ('Wheelchair', 152), ('Chair', 151), ('Boot', 149), ('Guitar', 144), ('Skateboard', 142), ('Football', 136), ('Necklace', 120), ('Baseball glove', 114), ('Surfboard', 110), ('Cake', 108), ('Tennis racket', 105), ('Paddle', 103), ('Motorcycle', 93), ('Scarf', 89), ('Trumpet', 87), ('Table tennis racket', 86), ('Swim cap', 82), ('Cart', 81), ('Racket', 81), ('French horn', 80), ('Violin', 80), ('Camera', 69), ('Coffee table', 66), ('Bed', 62), ('Beer', 61), ('Bus', 60), ('Volleyball (Ball)', 56), ('Cry', 55), ('Sushi', 55), ('Handbag', 53), ('Tiara', 53), ('Microphone', 52), ('Cello', 52), ('Trombone', 51), ('Baseball bat', 51), ('Crown', 48), ('Cocktail', 48), ('Juice', 46), ('Accordion', 43), ('Segway', 43), ('Saxophone', 43), ('Balance beam', 43), ('Billiard table', 42), ('Sombrero', 41), ('Piano', 41), ('Rifle', 40), ('Harp', 38), ('Flute', 38), ('Drum', 35), ('Musical keyboard', 33), ('Bicycle wheel', 32), ('Bottle', 31), ('Hiking equipment', 30), ('Earrings', 29), ('Rugby ball', 28), ('Bow and arrow', 25), ('Palm tree', 24), ('Rose', 24), ('Wine glass', 23), ('Golf cart', 23), ('Elephant', 22), ('Balloon', 22), ('Belt', 21), ('Oyster', 20), ('Infant bed', 19), ('Watermelon', 19), ('Ski', 17), ('Cutting board', 17), ('Book', 15), ('Salad', 15), ('Tripod', 15), ('Orange', 15), ('Shotgun', 14), ('Dumbbell', 14), ('Grape', 13), ('Strawberry', 13), ('Dog bed', 12), ('Coffee cup', 11), ('Harpsichord', 11), ('Mobile phone', 11), ('Gondola', 10), ('Cat', 10), ('Apple', 10), ('Truck', 10), ('Grapefruit', 10), ('Bowling equipment', 9), ('Horizontal bar', 9), ('Tennis ball', 9), ('Stationary bicycle', 9), ('Tent', 8), ('Airplane', 8), ('Watch', 8), ('Organ (Musical Instrument)', 8), ('Ladder', 8), ('Sofa bed', 7), ('Muffin', 7), ('Suitcase', 7), ('Unicycle', 6), ('Backpack', 6), ('Binoculars', 6), ('Van', 6), ('Doll', 6), ('Handgun', 5), ('Limousine', 5), ('Cake stand', 5), ('Lobster', 5), ('Ambulance', 5), ('Panda', 5), ('Cricket ball', 5), ('Countertop', 5), ('Stool', 4), ('Banjo', 4), ('Cat furniture', 4), ('Indoor rower', 4), ('Monkey', 4), ('Bench', 4), ('Cannon', 3), ('Helicopter', 3), ('Oboe', 3), ('Sword', 3), ('Tank', 3), ('Flying disc', 3), ('Teddy bear', 3), ('Washing machine', 3), ('Treadmill', 2), ('Box', 2), ('Ice cream', 2), ('Taxi', 2), ('Studio couch', 2), ('Train', 2), ('Carrot', 2), ('Candy', 2), ('Wok', 2), ('Common sunflower', 2), ('Sea lion', 2), ('Stethoscope', 2), ('Jet ski', 2), ('Pen', 2), ('Kitchen & dining room table', 2), ('Pomegranate', 2), ('Milk', 1), ('Plate', 1), ('Honeycomb', 1), ('Egg (Food)', 1), ('Lizard', 1), ('Loveseat', 1), ('Dolphin', 1), ('Whale', 1), ('Brown bear', 1), ('Picnic basket', 1), ('Plastic bag', 1), ('Punching bag', 1), ('Lemon', 1), ('Cheese', 1), ('Cupboard', 1), ('Personal flotation device', 1), ('Snowmobile', 1), ('Flowerpot', 1), ('Broccoli', 1), ('Cucumber', 1), ('Christmas tree', 1), ('Hamster', 1), ('Pasta', 1), ('Shark', 1), ('Kite', 1), ('Tart', 1), ('Pumpkin', 1), ('Crab', 1), ('Mug', 1), ('Dinosaur', 1), ('Tablet computer', 1), ('Bowl', 1)]
attributes_names = [name for name, count in attributes_counter]

for attr1, attr2 in ATTRIBUTE_TUPLES:
    assert attr1 in attributes_names
    assert attr2 in attributes_names


def show_image(image, regions_and_attributes=None):
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    response = requests.get(image.url)
    img = PIL_Image.open(io.BytesIO(response.content))
    plt.imshow(img)
    ax = plt.gca()
    for region, attribute in regions_and_attributes:
        ax.add_patch(Rectangle((region.x, region.y),
                               region.width,
                               region.height,
                               fill=False,
                               edgecolor='red',
                               linewidth=3))
        ax.text(region.x, region.y, region.names[0]  + f" ({attribute})", style='italic', bbox={'facecolor':'white', 'alpha':0.7, 'pad':10})
    plt.tick_params(labelbottom='off', labelleft='off')
    plt.show()


def show_image_pair(image_1_path, image_2_path, regions_and_attributes_1, regions_and_attributes_2):
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
        img_1_data_adjusted = img_1_data.crop((0, 0, img_1_data.width, img_2_data.height))
        img_2_data_adjusted = img_2_data
    else:
        img_1_data_adjusted = img_1_data
        img_2_data_adjusted = img_2_data.crop((0, 0, img_2_data.width, img_1_data.height))

    if img_1_data_adjusted.mode == "L":
        img_1_data_adjusted = img_1_data_adjusted.convert('L')
    image = np.column_stack((img_1_data_adjusted, img_2_data_adjusted))

    plt.imshow(image)
    ax = plt.gca()
    colors = ["green", "red"]
    for relationship, color in zip(regions_and_attributes_1, colors):
        bb = relationship.bounding_box
        ax.add_patch(Rectangle((bb[0] * img_1_data.width, bb[1] * img_1_data.height),
                               bb[2] * img_1_data.width,
                               bb[3] * img_1_data.height,
                               fill=False,
                               edgecolor=color,
                               linewidth=3))
        ax.text(bb[0] * img_1_data.width, bb[1] * img_1_data.height, relationship.Label1 + f" ({relationship.Label2})", style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})

    x_offset = img_1_data.width
    for relationship, color in zip(regions_and_attributes_2, colors):
        bb = relationship.bounding_box
        ax.add_patch(Rectangle((bb[0] * img_2_data.width + x_offset, bb[1] * img_2_data.height),
                               bb[2] * img_2_data.width,
                               bb[3] * img_2_data.height,
                               fill=False,
                               edgecolor=color,
                               linewidth=3))
        ax.text(bb[0] * img_2_data.width + x_offset, bb[1] * img_2_data.height, relationship.Label1 + f" ({relationship.Label2})", style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})

    plt.tick_params(labelbottom='off', labelleft='off')
    plt.show()


def is_subj_attr_in_image(sample, subject, attribute):
    if sample.relationships:
        for relationship in sample.relationships.detections:
            if relationship.Label1 == subject and relationship.Label2 == attribute:
                return relationship

    return False


def find_other_subj_with_attr(sample, subject, attribute):
    if sample.relationships:
        for relationship in sample.relationships.detections:
            if relationship.Label1 != subject and relationship.Label2 == attribute:
                # verify that they are not synonyms:
                if not {subject, relationship.Label1} in NOUN_SYNONYMS:
                    return relationship

    return None


def generate_eval_set_attribute_noun_dependencies(dataset, attribute_tuples):
    eval_set = {tuple: [] for tuple in attribute_tuples}

    for target_tuple in attribute_tuples:
        print("Looking for: ", target_tuple)
        target_attribute, distractor_attribute = target_tuple
        for sample_target in dataset:
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
                            relationship_visual_distractor = (
                                find_other_subj_with_attr(
                                    sample_target, target_subject, distractor_attribute
                                )
                            )
                            if relationship_visual_distractor:
                                # print("Found match: ", scene_graph_target.image)
                                # show_image(scene_graph_target.image, [(target_subject, target_attribute), (visual_distractor_subject, distractor_attribute)])

                                # print("Looking for counterexample image.. ")
                                for sample_distractor in dataset:

                                    # check that distractor IS in the distractor image:
                                    counterexample_relationship_target = is_subj_attr_in_image(
                                        sample_distractor, target_subject, distractor_attribute
                                    )
                                    if counterexample_relationship_target:
                                        #TODO: distractor subject == target subject?!

                                        # check that target IS NOT in same image:
                                        if not is_subj_attr_in_image(
                                            sample_distractor, target_subject, target_attribute
                                        ):

                                            # check that visual distractor IS in image
                                            counterexample_relationship_visual_distractor = (
                                                find_other_subj_with_attr(
                                                    sample_distractor, target_subject, target_attribute
                                                )
                                            )
                                            # TODO: enforce that distractor subjects are the same?
                                            if counterexample_relationship_visual_distractor: # and distractor_visual_distractor_subject.synsets[0].name == visual_distractor_subject.synsets[0].name:
                                                print(f"Found minimal pair: {sample_target.open_images_id} {sample_distractor.open_images_id}")
                                                # show_image_pair(sample_target.filepath, sample_distractor.filepath, [relationship_target, relationship_visual_distractor], [counterexample_relationship_target, counterexample_relationship_visual_distractor])

                                                # Add tuple of example and counter-example
                                                eval_set[target_tuple].append(({
                                                        "img_target": sample_target.filepath,
                                                        "img_distractor": sample_distractor.filepath,
                                                        "relationship_target": relationship_target,
                                                        "relationship_visual_distractor": relationship_visual_distractor,
                                                    },
                                                    {
                                                        "img_target": sample_distractor.filepath,
                                                        "img_distractor": sample_target.filepath,
                                                        "relationship_target": counterexample_relationship_target,
                                                        "relationship_visual_distractor": counterexample_relationship_visual_distractor,
                                                    }
                                                ))

        print(f"Found {len(eval_set[target_tuple])} examples for {target_tuple}.\n\n")

    return eval_set

if __name__ == "__main__":
    # relationships = pd.read_csv("data/relationships_test_display_names.csv")
    # most_common_attributes = Counter(relationships["LabelName2"].values).most_common()

    # dataset = foz.load_zoo_dataset(
    #     "open-images-v6",
    #     split="test",
    #     label_types=["relationships"],
    #     max_samples=5000,
    # )

    # session = fiftyone.launch_app(dataset)
    # session.view = dataset.sort_by("open_images_id").limit(10)
    # session.wait()

    # eval_sets = generate_eval_set_attribute_noun_dependencies(dataset, ATTRIBUTE_TUPLES)
    #
    # pickle.dump(eval_sets, open("data/attribute_noun.p", "wb"))

    eval_sets = pickle.load(open("data/attribute_noun.p", "rb"))

    for tuple, eval_set in eval_sets.items():
        if len(eval_set) > 0:
            print(f"{tuple} ({len(eval_set)} examples)")
            # for example, counterexample in eval_set:
            #     show_image_pair(example["img_target"], example["img_distractor"], [example["relationship_target"], example["relationship_visual_distractor"]],
            #                     [counterexample["relationship_target"], counterexample["relationship_visual_distractor"]])
