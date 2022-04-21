import json
import os
import pickle

from utils import SYNONYMS, SUBJECT, REL, OBJECT


def make_rel_serializable(relationship):
    if isinstance(relationship, dict):
        relationship = {
            "id": relationship["id"],
            "Label1": relationship["Label1"],
            "label": relationship["label"],
            "Label2": relationship["Label2"],
            "bounding_box": list(relationship["bounding_box"]),
        }
    else:
        relationship = {
            "id": relationship.id,
            "Label1": relationship.Label1,
            "label": relationship.label,
            "Label2": relationship.Label2,
            "bounding_box": list(relationship.bounding_box),
        }
    return relationship


if __name__ == "__main__":
    data_json = []
    id = 0
    for split in ["train", "test"]:
        for pos in ["object", "subject"]:
            file = f'filtered_eval_sets/{pos}-{split}.p'
            print("Processing: ", file)
            data = pickle.load(open(file, "rb"))
            for key, samples in data.items():
                for sample in samples:
                    example = {
                        "id": id,
                        "img_filename": os.path.basename(sample["img_example"]),
                        "sentence_target": sample["sentence_target"],
                        "sentence_distractor": sample["sentence_distractor"],
                        "relationship_target": make_rel_serializable(sample["relationship_target"]),
                        "relationship_distractor": make_rel_serializable(sample["relationship_visual_distractor"]),
                        "subject": SYNONYMS[sample["relationship_target"][SUBJECT]][0],
                        "predicate": SYNONYMS[sample["relationship_target"][REL]][0],
                        "object": SYNONYMS[sample["relationship_target"][OBJECT]][0],
                        "word_target": key[0],
                        "word_distractor": key[1],
                        "pos": pos,
                        "open_images_split": split,
                    }
                    id += 1
                    data_json.append(example)
                    counterexample = {
                        "id": id,
                        "img_filename": os.path.basename(sample["img_counterexample"]),
                        "sentence_target": sample["sentence_distractor"],
                        "sentence_distractor": sample["sentence_target"],
                        "relationship_target": make_rel_serializable(sample["counterexample_relationship_target"]),
                        "relationship_distractor": make_rel_serializable(sample["counterexample_relationship_visual_distractor"]),
                        "subject": SYNONYMS[sample["counterexample_relationship_target"][SUBJECT]][0],
                        "predicate": SYNONYMS[sample["counterexample_relationship_target"][REL]][0],
                        "object": SYNONYMS[sample["counterexample_relationship_target"][OBJECT]][0],
                        "word_target": key[1],
                        "word_distractor": key[0],
                        "pos": pos,
                        "open_images_split": split,
                    }
                    id += 1
                    data_json.append(counterexample)

    with open(f'filtered_eval_sets/eval_set.json', 'w') as file:
        json.dump(data_json, file)

