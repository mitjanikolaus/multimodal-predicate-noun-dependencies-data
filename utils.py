import visual_genome.local as vg

DATA_DIR = '../visual_genome_python_driver/visual_genome/data/'
IMAGE_DATA_DIR = '../visual_genome_python_driver/visual_genome/data/by-id/'

NOUN_PERSON = "person.n.01"
NOUN_MAN = "man.n.01"
NOUN_WOMAN = "woman.n.01"
NOUN_CHILD = "child.n.01" #TODO: male_child?

REL_WATCH = "watch.v.01"
REL_WEAR = "wear.v.01"
REL_HAVE = "have.v.01"


def filter_scene_graphs():
    # region_descriptions = vg.get_all_region_descriptions(data_dir=DATA_DIR)
    # for desc_image in region_descriptions:
    #     for desc in desc_image:
    #         #TODO check for minimum region size?
    #         print(desc.phrase)

    rels_men = []
    rels_women = []

    print("Loading scene graphs..")
    scene_graphs = vg.get_scene_graphs(start_index=0, end_index=1000, data_dir=DATA_DIR, image_data_dir=IMAGE_DATA_DIR)
    print("Done loading scene graphs.")

    for scene_graph in scene_graphs:
        for relationship in scene_graph.relationships:
            # print(f"{relationship.predicate}")
            # print(f"{relationship.subject} {relationship.predicate} {relationship.object}")

            subject_synsets = [s.name for s in relationship.subject.synsets]
            object_synsets = [s.name for s in relationship.object.synsets]
            relationship_synsets = [s.name for s in relationship.synset]

            if NOUN_MAN in subject_synsets and REL_WEAR in relationship_synsets:
                rels_men.append(relationship)
            if NOUN_CHILD in subject_synsets and REL_WEAR in relationship_synsets:
                rels_women.append(relationship)

    print(rels_women)
    print(rels_men)
    for rel_man in rels_men:
        for rel_woman in rels_women:
            if rel_man.object.names[0] == rel_woman.object.names[0]: #TODO only first names?
                print(rel_man)
                print(rel_woman)

    #TODO: make sure distractor is not in same image!


if __name__ == '__main__':
    filter_scene_graphs()

