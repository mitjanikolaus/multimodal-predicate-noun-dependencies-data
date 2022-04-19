import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from utils import multiply_df_for_per_word_analyses, OBJECTS_VERBS

NOUNS = ["Woman", "Man", "Girl", "Boy"]

MIN_NUM_TEST_SAMPLES = 10


def eval_set_stats(args):
    print("Loading: ", args.input_file)
    eval_set = pd.read_json(args.input_file)
    eval_set_per_word = multiply_df_for_per_word_analyses(eval_set)

    words_enough_samples = [k for k, v in eval_set_per_word.groupby("word").size().to_dict().items() if v >= MIN_NUM_TEST_SAMPLES]
    eval_set_per_word = eval_set_per_word[eval_set_per_word.word.isin(words_enough_samples)]

    _, axes = plt.subplots(2, 1, figsize=(4, 6), sharex="none", gridspec_kw={'height_ratios': [1, 4]})

    data_nouns = eval_set_per_word[eval_set_per_word.word.isin(NOUNS)]
    g = data_nouns.groupby("word").size().plot.barh(color="black", ax=axes[0])
    g.set_xscale("log")
    g.set_xticks([1, 10, 100, 1000])
    g.set_xticklabels(["", 10, 100, 1000])
    plt.xlabel("# Samples")
    plt.ylabel("")
    axes[0].set_title("Nouns")
    axes[0].xaxis.label.set_visible(False)
    axes[0].yaxis.label.set_visible(False)

    data_predicates = eval_set_per_word[~eval_set_per_word.word.isin(NOUNS)]

    full_predicates = {}
    predicates = data_predicates.word.unique()
    for predicate in predicates:
        print(predicate)
        full_pred = eval_set[eval_set.word_target == predicate].iloc[0].sentence_target.split("is ")[1]
        full_predicates[predicate] = full_pred
    data_predicates.word.replace(full_predicates, inplace=True)

    g = data_predicates.groupby("word").size().plot.barh(color="black", ax=axes[1])
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
