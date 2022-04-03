import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from utils import multiply_df_for_per_word_analyses, OBJECTS_VERBS


def eval_set_stats(args):
    print("Loading: ", args.input_file)
    eval_set = pd.read_json(args.input_file)
    eval_set_per_word = multiply_df_for_per_word_analyses(eval_set)

    g = eval_set_per_word.groupby("word").size().plot.bar()
    plt.xticks(rotation=75)
    plt.subplots_adjust(bottom=0.3)
    g.set_yscale("log")
    plt.ylabel("# Samples")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig("results_figures/num_samples_per_word.png")

    data_nouns = eval_set_per_word[~eval_set_per_word.word.isin(OBJECTS_VERBS)]
    g = data_nouns.groupby("word").size().plot.bar()
    plt.xticks(rotation=75)
    plt.subplots_adjust(bottom=0.3)
    g.set_yscale("log")
    plt.ylabel("# Samples")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig("results_figures/num_samples_per_word_nouns.png")

    data_verbs = eval_set_per_word[eval_set_per_word.word.isin(OBJECTS_VERBS)]
    g = data_verbs.groupby("word").size().plot.bar()
    plt.xticks(rotation=75)
    plt.subplots_adjust(bottom=0.3)
    g.set_yscale("log")
    plt.ylabel("# Samples")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig("results_figures/num_samples_per_word_verbs.png")

    plt.figure(figsize=(20, 10))
    g = eval_set.groupby(["word_target", "word_distractor"]).size().plot.bar()
    g.set_yscale("log")
    plt.xticks(rotation=75)
    plt.subplots_adjust(bottom=0.3)
    plt.ylabel("# Samples")
    plt.tight_layout()
    plt.savefig("results_figures/num_samples_target_distractor.png")

    plt.figure(figsize=(20, 10))
    g = sns.countplot(data=eval_set, x="subject", hue="object")
    g.set_yscale("log")
    # plt.xticks(rotation=75)
    # plt.subplots_adjust(bottom=0.3)
    plt.ylabel("# Samples")
    plt.tight_layout()
    plt.savefig("results_figures/num_samples_subject_object.png")

    plt.show()


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input-file", type=str, required=True)

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    eval_set_stats(args)
