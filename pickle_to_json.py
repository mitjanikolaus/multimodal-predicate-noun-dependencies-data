import json
import os
import pickle


def make_rel_serializable(relationship):
    relationship = {
        "id": relationship["id"],
        "Label1": relationship["Label1"],
        "label": relationship["label"],
        "Label2": relationship["Label2"],
        "bounding_box": list(relationship["bounding_box"]),
    }
    return relationship


if __name__ == "__main__":
    for pos in ["object", "subject"]:
        data = pickle.load(open(f'filtered_eval_sets/{pos}-train.p', "rb"))
        data_json = []
        id = 0
        for key, samples in data.items():
            for sample in samples:
                example = {
                    "id": id,
                    "img_filename": os.path.basename(sample["img_example"]),
                    "sentence_target": sample["sentence_target"],
                    "sentence_distractor": sample["sentence_distractor"],
                    "relationship_target": make_rel_serializable(sample["relationship_target"]),
                    "relationship_distractor": make_rel_serializable(sample["relationship_visual_distractor"]),
                    "word_target": key[0],
                    "word_distractor": key[1],
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
                    "word_target": key[1],
                    "word_distractor": key[0],
                }
                id += 1
                data_json.append(counterexample)

        with open(f'filtered_eval_sets/{pos}-train.json', 'w') as file:
            json.dump(data_json, file)

