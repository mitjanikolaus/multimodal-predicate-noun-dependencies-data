import argparse
import os
import pickle

from utils import get_target_and_distractor_sentence, get_file_name_of_cropped_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=str)
    return parser.parse_args()


def eval_2afc(model, tokenizer, img_features, classification_score_function, args):
    """Evaluate model using 2-alternative forced choice task."""

    eval_sets = pickle.load(open(args.eval_set, "rb"))

    successes = 0
    failures = 0

    for key, set in eval_sets.items():
        print(key)
        for sample in set:
            text_target, text_distractor = get_target_and_distractor_sentence(sample)

            if "cropped" in args.img_features_path:
                key_example = get_file_name_of_cropped_image(sample["img_example"], sample["relationship_target"])
                example_features = img_features[key_example]
                key_counterexample = get_file_name_of_cropped_image(sample["img_counterexample"], sample["counterexample_relationship_target"])
                counterexample_features = img_features[key_counterexample]
            else:
                example_features = img_features[os.path.basename(sample['img_example'])]
                counterexample_features = img_features[os.path.basename(sample['img_counterexample'])]

            prob_target_match = classification_score_function(model, tokenizer, text_target, example_features, text_target)
            prob_distractor_match = classification_score_function(model, tokenizer, text_distractor, example_features, text_target)

            if prob_target_match > prob_distractor_match:
                result_example = "SUCCESS"
                successes += 1
            else:
                result_example = "FAILURE"
                failures += 1

            result_example += f" ({prob_target_match:.3f} vs. {prob_distractor_match:.3f})"

            # Counterexample: switch target and distractor sentence
            text_distractor, text_target = text_target, text_distractor

            prob_target_match = classification_score_function(model, tokenizer, text_target, counterexample_features, text_target)
            prob_distractor_match = classification_score_function(model, tokenizer, text_distractor, counterexample_features, text_target)

            if prob_target_match > prob_distractor_match:
                result_counterexample = "SUCCESS"
                successes += 1
            else:
                result_counterexample = "FAILURE"
                failures += 1

            result_counterexample += f" ({prob_target_match:.3f} vs. {prob_distractor_match:.3f})"

            # show_sample(sample, text_distractor, text_target, result_example, result_counterexample)

    print(f"Accuracy: {round(100*successes/(successes+failures), 2)} %")
