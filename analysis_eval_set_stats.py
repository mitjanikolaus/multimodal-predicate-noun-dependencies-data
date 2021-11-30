import argparse
import pickle

import matplotlib.pyplot as plt


from utils import convert_eval_sets_to_one_sample_per_row


def eval_set_stats(args):
    print("Processing: ", args.input_file)
    eval_sets = pickle.load(open(args.input_file, "rb"))

    eval_set = convert_eval_sets_to_one_sample_per_row(eval_sets)

    eval_set.groupby("target_word").size().plot.bar()
    plt.xticks(rotation=75)
    plt.subplots_adjust(bottom=0.3)
    plt.ylabel("# Samples")

    plt.figure()
    eval_set.groupby(["target_word", "distractor_word"]).size().plot.bar()
    plt.xticks(rotation=75)
    plt.subplots_adjust(bottom=0.3)
    plt.ylabel("# Samples")
    plt.show()


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(type=str, dest="input_file")

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    eval_set_stats(args)
