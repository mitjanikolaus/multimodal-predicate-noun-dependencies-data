import os
import pickle


from tqdm import tqdm

from utils import get_local_image_path
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arg_values = get_args()

    eval_sets = pickle.load(open(arg_values.eval_set, "rb"))

    img_paths = []

    for key, set in eval_sets.items():
        print(key)
        for sample in tqdm(set):

            img_example_path = get_local_image_path(sample["img_example"])
            img_counterexample_path = get_local_image_path(sample["img_counterexample"])
            img_paths.append(img_example_path)
            img_paths.append(img_counterexample_path)

    out_file_name = "img_paths_" + os.path.basename(arg_values.eval_set.split(".p")[0]) + ".txt"
    with open(out_file_name, "w") as text_file:
        for path in img_paths:
            text_file.write(path+"\n")
