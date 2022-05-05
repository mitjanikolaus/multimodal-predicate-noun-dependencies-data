import argparse

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from utils import multiply_df_for_per_concept_analyses, transform_to_per_pair_eval_set

NOUNS = ["Woman", "Man", "Girl", "Boy"]

MIN_NUM_TEST_SAMPLES = 0


def eval_set_stats(args):
    print("Loading: ", args.input_file)
    eval_set = pd.read_json(args.input_file)
    eval_set_pairs = transform_to_per_pair_eval_set(eval_set)
    eval_set_per_concept = multiply_df_for_per_concept_analyses(eval_set_pairs)

    words_enough_samples = [k for k, v in eval_set_per_concept.groupby("concept").size().to_dict().items() if v >= MIN_NUM_TEST_SAMPLES]
    eval_set_per_concept = eval_set_per_concept[eval_set_per_concept.concept.isin(words_enough_samples)]

    _, axes = plt.subplots(2, 1, figsize=(4, 7), sharex="none", gridspec_kw={'height_ratios': [1, 8]})

    data_nouns = eval_set_per_concept[eval_set_per_concept.concept.isin(NOUNS)]
    g = data_nouns.groupby("concept").size().plot.barh(color="black", ax=axes[0])
    g.set_xscale("log")
    g.set_xticks([1, 10, 100, 1000])
    g.set_xticklabels(["", 10, 100, 1000])
    plt.xlabel("# Samples")
    plt.ylabel("")
    axes[0].set_title("Nouns")
    axes[0].xaxis.label.set_visible(False)
    axes[0].yaxis.label.set_visible(False)

    data_predicates = eval_set_per_concept[~eval_set_per_concept.concept.isin(NOUNS)].copy()

    full_predicates = {}
    predicates = data_predicates.concept.unique()
    for predicate in predicates:
        full_pred = eval_set[eval_set.object == predicate].iloc[0].sentence_target.split("is ")[1]
        full_predicates[predicate] = full_pred
    data_predicates.concept.replace(full_predicates, inplace=True)

    g = data_predicates.groupby("concept").size().plot.barh(color="black", ax=axes[1])
    g.set_xscale("log")
    g.set_xticks([1, 10, 100, 1000])
    g.set_xticklabels(["", 10, 100, 1000])

    plt.xlabel("# Samples")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("results_figures/num_samples_per_concept.pdf", dpi=300)

    plt.figure(figsize=(20, 10))
    g = eval_set.groupby(["word_target", "word_distractor"]).size().plot.bar()
    g.set_yscale("log")
    plt.xticks(rotation=75)
    plt.subplots_adjust(bottom=0.3)
    plt.ylabel("# Samples")
    plt.tight_layout()
    plt.savefig("results_figures/num_samples_target_distractor.pdf", dpi=300)

    plt.figure(figsize=(20, 10))
    g = sns.countplot(data=eval_set, x="subject", hue="object")
    g.set_yscale("log")
    plt.ylabel("# Samples")
    plt.tight_layout()
    plt.savefig("results_figures/num_samples_subject_object.pdf", dpi=300)

    # plt.show()


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input-file", type=str, required=True)

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    eval_set_stats(args)
