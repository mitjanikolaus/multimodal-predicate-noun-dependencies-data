import os
from collections import Counter
import pandas as pd
from tqdm import tqdm


def get_most_common_attributes():
    """Get the most common attributes from the open images test set"""
    relationship_names = pd.read_csv("data/oidv6-class-descriptions.csv")
    attribute_names_new = pd.read_csv(
        "data/oidv6-attributes-description_new.csv",
        names=["LabelName", "DisplayName"],
        header=None,
    )
    attribute_names_overlap = pd.read_csv(
        "data/oidv6-attributes-description_overlap.csv",
        names=["LabelName", "DisplayName"],
        header=None,
    )

    relationship_names.set_index("LabelName", inplace=True)
    attribute_names_new.set_index("LabelName", inplace=True)
    attribute_names_overlap.set_index("LabelName", inplace=True)

    relationship_names.update(attribute_names_overlap)
    relationship_names = relationship_names.append(attribute_names_new)

    relationships = pd.read_csv(os.path.expanduser("~/data/open_images/oidv6-train-annotations-vrd.csv"))
    relationships.reset_index(drop=True, inplace=True)

    relationships["LabelName1"] = relationships.apply(
        lambda x: relationship_names.loc[x[1].strip()].DisplayName, axis=1
    )
    relationships["LabelName2"] = relationships.apply(
        lambda x: relationship_names.loc[x[2].strip()].DisplayName, axis=1
    )

    subjects_counter = Counter(relationships["LabelName1"].values)
    print(subjects_counter.most_common())
    pd.DataFrame(subjects_counter.most_common()).to_csv("data/subject_occurrences.csv", header=False, index=False)

    obj_counter = Counter(relationships["LabelName2"].values)
    print(obj_counter.most_common())
    pd.DataFrame(obj_counter.most_common()).to_csv("data/obj_occurrences.csv", header=False, index=False)

    rel_counter = Counter(relationships["RelationshipLabel"].values)
    print(rel_counter.most_common())
    pd.DataFrame(rel_counter.most_common()).to_csv("data/rel_occurrences.csv", header=False, index=False)

    attributes_persons = Counter(
        relationships[
            relationships.LabelName1.isin(["Man", "Woman", "Boy", "Girl"])
        ].LabelName2.values
    ).most_common()
    print(f"Most common attributes for persons: {attributes_persons}")

    relationships.to_csv("results/relationships_train_display_names.csv")

    # Find nouns that can be both subject and object of a relationship
    print("Looking for nouns that can be both subject and object: ")
    subj_and_obj = set(relationships["LabelName2"].values) & set(
        relationships["LabelName1"].values
    )
    subj_and_obj_filtered = set()
    for noun in tqdm(subj_and_obj):
        rel_view_1 = relationships[relationships["LabelName1"] == noun]
        rel_view_2 = relationships[relationships["LabelName2"] == noun]
        overlapping_rel = set(rel_view_1["RelationshipLabel"].values) & set(
            rel_view_2["RelationshipLabel"].values
        )
        if overlapping_rel:
            print(noun)
            print(overlapping_rel)
            subj_and_obj_filtered.add(noun)

    print("Summary: ", subj_and_obj_filtered)


if __name__ == "__main__":
    get_most_common_attributes()
