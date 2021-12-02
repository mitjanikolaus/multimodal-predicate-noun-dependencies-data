import argparse
import pickle

import matplotlib.pyplot as plt
import pandas as pd

from utils import convert_eval_sets_to_one_sample_per_row, SUBJECT, OBJECT


def eval_set_stats(args):
    all_eval_sets = []
    for input_file in args.input_files:
        print("Loading: ", input_file)
        eval_set = pickle.load(open(input_file, "rb"))
        eval_set = convert_eval_sets_to_one_sample_per_row(eval_set)

        all_eval_sets.append(eval_set)

    eval_sets = pd.concat(all_eval_sets, ignore_index=True)

    eval_sets.groupby("target_word").size().plot.bar()
    plt.xticks(rotation=75)
    plt.subplots_adjust(bottom=0.3)
    plt.ylabel("# Samples")
    plt.tight_layout()
    plt.savefig("results_figures/num_samples_target_word.png")

    plt.figure(figsize=(20, 10))
    eval_sets.groupby(["target_word", "distractor_word"]).size().plot.bar()
    plt.xticks(rotation=75)
    plt.subplots_adjust(bottom=0.3)
    plt.ylabel("# Samples")
    plt.tight_layout()
    plt.savefig("results_figures/num_samples_target_distractor.png")

    plt.figure(figsize=(20, 10))
    eval_sets["subject"] = [r[SUBJECT] for r in eval_sets["relationship_target"]]
    eval_sets["object"] = [r[OBJECT] for r in eval_sets["relationship_target"]]
    eval_sets["subj_obj"] = [(t, d) for t, d in zip(eval_sets["subject"], eval_sets["object"])]
    eval_sets.groupby("subj_obj").size().plot.bar()
    plt.xticks(rotation=75)
    plt.subplots_adjust(bottom=0.3)
    plt.ylabel("# Samples")
    plt.tight_layout()
    plt.savefig("results_figures/num_samples_subject_object.png")

    plt.show()


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(type=str, dest="input_files", nargs="+")

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    eval_set_stats(args)
