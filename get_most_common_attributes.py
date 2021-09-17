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


def get_most_common_attributes():
    """Get the most common attributes from the open images test set"""
    relationship_names = pd.read_csv("data/oidv6-class-descriptions.csv")
    attribute_names_new = pd.read_csv("data/oidv6-attributes-description_new.csv", names=["LabelName", "DisplayName"],
                                  header=None)
    attribute_names_overlap = pd.read_csv("data/oidv6-attributes-description_overlap.csv", names=["LabelName", "DisplayName"],
                                  header=None)

    relationship_names.set_index("LabelName", inplace=True)
    attribute_names_new.set_index("LabelName", inplace=True)
    attribute_names_overlap.set_index("LabelName", inplace=True)

    relationship_names.update(attribute_names_overlap)
    relationship_names = relationship_names.append(attribute_names_new)

    relationships = pd.read_csv("data/oidv6-test-annotations-vrd.csv")
    relationships.reset_index(drop=True, inplace=True)

    relationships["LabelName1"] = relationships.apply(lambda x: relationship_names.loc[x[1].strip()].DisplayName,
                                                      axis=1)
    relationships["LabelName2"] = relationships.apply(lambda x: relationship_names.loc[x[2].strip()].DisplayName,
                                                      axis=1)

    print(Counter(relationships["LabelName1"].values).most_common())
    print(Counter(relationships["LabelName2"].values).most_common())
    print(Counter(relationships["RelationshipLabel"].values).most_common())

    relationships.to_csv("data/relationships_test_display_names.csv")



if __name__ == "__main__":
    get_most_common_attributes()
