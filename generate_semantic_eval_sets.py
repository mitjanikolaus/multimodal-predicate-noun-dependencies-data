import os

import visual_genome.local as vg_local
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image as PIL_Image
from matplotlib.patches import Rectangle
import io
import requests

DATA_DIR = "../visual_genome_python_driver/visual_genome/data/"
IMAGE_DATA_DIR = "../visual_genome_python_driver/visual_genome/data/by-id/"

SPLIT_TEST = "data/visual_genome_splits/test.txt"

NOUN_PERSON = "person.n.01"
NOUN_MAN = "man.n.01"
NOUN_WOMAN = "woman.n.01"
NOUN_CHILD = "child.n.01"  # TODO: male_child?

REL_WATCH = "watch.v.01"
REL_WEAR = "wear.v.01"
REL_HAVE = "have.v.01"


ATTRIBUTE_TUPLES = [
    ("standing", "sitting"),
    ("standing", "walking"),
    ("happy", "sad"),
    ("happy", "angry"),
    # ("happy", "upset"),
    # ("happy", "scared"),
    # ("happy", "mad"),
    # ("happy", "afraid"),
    # ("happy", "surprised"),
]


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


def show_image_pair(image_1, image_2, regions_and_attributes_1, regions_and_attributes_2):

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    img_1_data = PIL_Image.open(io.BytesIO(requests.get(image_1.url).content))
    img_2_data = PIL_Image.open(io.BytesIO(requests.get(image_2.url).content))

    # Make images equal size:
    if img_2_data.height > img_1_data.height:
        img_1_data = img_1_data.crop((0, 0, img_2_data.width, img_2_data.height))
    else:
        img_2_data = img_2_data.crop((0, 0, img_1_data.width, img_1_data.height))

    image = np.column_stack((img_1_data, img_2_data))

    plt.imshow(image)
    ax = plt.gca()
    for region, attribute in regions_and_attributes_1:
        ax.add_patch(Rectangle((region.x, region.y),
                               region.width,
                               region.height,
                               fill=False,
                               edgecolor='green',
                               linewidth=3))
        ax.text(region.x, region.y, region.names[0] + f" ({attribute})", style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})

    x_offset = img_1_data.width
    for region, attribute in regions_and_attributes_2:
        ax.add_patch(Rectangle((region.x + x_offset, region.y),
                               region.width,
                               region.height,
                               fill=False,
                               edgecolor='red',
                               linewidth=3))
        ax.text(region.x + x_offset, region.y, region.names[0] + f" ({attribute})", style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})

    plt.tick_params(labelbottom='off', labelleft='off')
    plt.show()


def check_is_subj_attr_in_scene_graph(scene_graph, subject, attribute):
    for relationship in scene_graph.relationships:
        subject_synset = (
            relationship.subject.synsets[0].name
            if relationship.subject.synsets
            else None
        )
        subject_attributes = (
            relationship.subject.attributes if relationship.subject.synsets else []
        )

        if subject.synsets[0].name == subject_synset and attribute in subject_attributes:
            return relationship.subject

    return False


def find_other_subj_with_attr(scene_graph, subject, attribute):
    for relationship in scene_graph.relationships:
        subject_synset = (
            relationship.subject.synsets[0].name
            if relationship.subject.synsets
            else None
        )
        subject_attributes = (
            relationship.subject.attributes if relationship.subject.synsets else []
        )

        if subject.synsets[0].name != subject_synset and attribute in subject_attributes:
            return relationship.subject

    return None


def generate_eval_set_attribute_noun_dependencies(scene_graphs, attribute_tuples):
    #TODO: do the same but exchange subject?

    eval_set = []

    for target_tuple in attribute_tuples:
        print("Looking for: ", target_tuple)
        target_attribute, distractor_attribute = target_tuple
        for scene_graph_target in scene_graphs:
            for relationship_target in scene_graph_target.relationships:

                subject_attributes = (
                    relationship_target.subject.attributes
                    if relationship_target.subject.synsets
                    else []
                )

                # TODO: check for attribute synonyms!
                if target_attribute in subject_attributes:
                    target_subject = relationship_target.subject

                    # check that distractor IS NOT in same image:
                    distractor_in_target_image = check_is_subj_attr_in_scene_graph(
                        scene_graph_target, target_subject, distractor_attribute
                    )
                    if not distractor_in_target_image:

                        # check that visual distractor IS in image
                        visual_distractor_subject = (
                            find_other_subj_with_attr(
                                scene_graph_target, target_subject, distractor_attribute
                            )
                        )
                        if visual_distractor_subject:
                            print("Found match: ", scene_graph_target.image)
                            # show_image(scene_graph_target.image, [(target_subject, target_attribute), (visual_distractor_subject, distractor_attribute)])

                            print("Looking for distractor image.. ")
                            for scene_graph_distractor in scene_graphs:

                                # check that distractor IS in the distractor image:
                                distractor_subject = check_is_subj_attr_in_scene_graph(
                                    scene_graph_distractor, target_subject, distractor_attribute
                                )

                                if distractor_subject != False:

                                    # check that target IS NOT in same image:
                                    target_in_distractor_image = check_is_subj_attr_in_scene_graph(
                                        scene_graph_distractor, distractor_subject, target_attribute
                                    )

                                    if not target_in_distractor_image:

                                        # check that visual distractor IS in image
                                        distractor_visual_distractor_subject = (
                                            find_other_subj_with_attr(
                                                scene_graph_distractor, distractor_subject, target_attribute
                                            )
                                        )
                                        # TODO: enforce that distractor subjects are the same?
                                        if distractor_visual_distractor_subject: # and distractor_visual_distractor_subject.synsets[0].name == visual_distractor_subject.synsets[0].name:
                                            print("Found distractor image: ", scene_graph_distractor.image)
                                            show_image_pair(scene_graph_target.image, scene_graph_distractor.image, [(target_subject, target_attribute), (visual_distractor_subject, distractor_attribute)], [(distractor_subject, distractor_attribute), (distractor_visual_distractor_subject, target_attribute)])

                                            # Add example
                                            eval_set.append({
                                                "img_target": scene_graph_target.image.id,
                                                "img_distractor": scene_graph_distractor.image.id,
                                                "target_subject": target_subject,
                                                "target_attribute": target_attribute,
                                                "visual_distractor_subject": visual_distractor_subject,
                                                "visual_distractor_attribute": distractor_attribute,
                                            })

                                            # Add counter-example
                                            eval_set.append({
                                                "img_target": scene_graph_distractor.image.id,
                                                "img_distractor": scene_graph_target.image.id,
                                                "target_subject": distractor_subject,
                                                "target_attribute": distractor_attribute,
                                                "visual_distractor_subject": distractor_visual_distractor_subject,
                                                "visual_distractor_attribute": target_attribute,
                                            })


    return eval_set


def read_split_ids_from_file(path):
    image_ids = []
    with open(path) as f:
        for line in f.readlines():
            image_id = int(line.split(' ')[0].split("/")[1].split(".")[0])
            image_ids.append(image_id)

    print(f"Loaded {len(image_ids)} image ids from {path}.")
    return image_ids


def get_scene_graphs_by_ids(ids_list,
                     data_dir='data/', image_data_dir='data/by-id/',
                     min_rels=0, max_rels=100):
    """
    Get scene graphs given locally stored .json files;
    requires `save_scene_graphs_by_id`.

    ids_list : get scene graphs listed by image ids
    data_dir : directory with `image_data.json` and `synsets.json`
    image_data_dir : directory of scene graph jsons saved by image id
                   (see `save_scene_graphs_by_id`)
    min_rels, max_rels: only get scene graphs with at least / less
                      than this number of relationships
    """
    images = {img.id: img for img in vg_local.get_all_image_data(data_dir) if img.id in ids_list}
    scene_graphs = []

    img_fnames = os.listdir(image_data_dir)

    for fname in img_fnames:
        image_id = int(fname.split('.')[0])
        if image_id in ids_list:
            scene_graph = vg_local.get_scene_graph(
                image_id, images, image_data_dir, data_dir + 'synsets.json')
            n_rels = len(scene_graph.relationships)
            if (min_rels <= n_rels <= max_rels):
                scene_graphs.append(scene_graph)

    return scene_graphs


if __name__ == "__main__":

    image_ids = read_split_ids_from_file(SPLIT_TEST)

    print("Loading scene graphs.. ", end="")
    scene_graphs = get_scene_graphs_by_ids(
        image_ids,
        data_dir=DATA_DIR,
        image_data_dir=IMAGE_DATA_DIR,
    )
    print(f"done loading {len(scene_graphs)} scene graphs.")

    eval_set = generate_eval_set_attribute_noun_dependencies(scene_graphs, ATTRIBUTE_TUPLES)

    print(eval_set)
    print(f"Found {len(eval_set)} examples.")
