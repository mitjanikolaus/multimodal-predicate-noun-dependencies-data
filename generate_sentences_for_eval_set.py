import argparse
import json
import pickle
from collections import Counter
import pandas as pd
import spacy

from utils.utils import (
    SYNONYMS,
    SUBJECT,
    OBJECT,
    OBJECTS_VERBS,
    OBJECTS_PLURAL,
    OBJECTS_TEXTURES,
    REL,
)


def generate_sentence_for_eval_set(args):
    nlp = spacy.load("en_core_web_sm")

    # Training data downloaded from https://ai.google.com/research/ConceptualCaptions/download
    data = pd.read_csv(
        "data/conceptual_captions/Train_GCC-training.tsv", sep="\t", header=None
    )
    captions = data[0].values

    eval_set = json.load(open(args.eval_set, "rb"))

    predicates = {}
    distinct_sentences = set()

    for id, sample in enumerate(eval_set):
        # Start from examples, which are on even indices
        if id % 2 == 0:
            original_subject = SYNONYMS[sample["relationship_target"][SUBJECT]][0]
            subject = original_subject.lower()

            original_object = SYNONYMS[sample["relationship_target"][OBJECT]][0]
            obj = transform_label(original_object)

            predicate = SYNONYMS[sample["relationship_target"][REL]][0]

            # Counterexample
            id_counterexample = id+1
            counterexample = eval_set[id_counterexample]
            distractor_original_subj = SYNONYMS[
                counterexample["relationship_target"][SUBJECT]
            ][0]
            distractor_subject = distractor_original_subj.lower()

            distractor_original_obj = SYNONYMS[
                counterexample["relationship_target"][OBJECT]
            ][0]
            distractor_obj = transform_label(distractor_original_obj)

            distractor_predicate = SYNONYMS[
                counterexample["relationship_target"][REL]
            ][0]

            if (original_subject, original_object) not in predicates or (
                distractor_original_subj,
                distractor_original_obj,
            ) not in predicates:
                predicates_target = get_predicates(
                    captions, nlp, subject, predicate, obj, original_object
                )
                predicates_distractor = get_predicates(
                    captions,
                    nlp,
                    distractor_subject,
                    distractor_predicate,
                    distractor_obj,
                    distractor_original_obj,
                )
                if subject == distractor_subject:
                    # Same subject -> we can have different predicates
                    predicate = predicates_target[0]
                    distractor_predicate = predicates_distractor[0]
                else:
                    # Same objects -> we want the same predicates
                    predicate = get_common_predicate(
                        predicates_target, predicates_distractor
                    )
                    distractor_predicate = predicate

                if predicate.endswith("ing"):
                    predicate = "is " + predicate
                if distractor_predicate.endswith("ing"):
                    distractor_predicate = "is " + distractor_predicate

                predicates[(original_subject, original_object)] = predicate
                predicates[
                    (distractor_original_subj, distractor_original_obj)
                ] = distractor_predicate

            sentence_target = generate_sentence_from_triplet(
                subject,
                predicates[(original_subject, original_object)],
                obj,
                original_object,
            )
            sample["sentence_target"] = sentence_target
            counterexample["sentence_distractor"] = sentence_target
            sentence_target_alt = generate_alternative_sentence(sentence_target)
            sample["sentence_target_alt"] = sentence_target_alt
            counterexample["sentence_distractor_alt"] = sentence_target_alt

            sentence_distractor = generate_sentence_from_triplet(
                distractor_subject,
                predicates[(distractor_original_subj, distractor_original_obj)],
                distractor_obj,
                distractor_original_obj,
            )
            sample["sentence_distractor"] = sentence_distractor
            counterexample["sentence_target"] = sentence_distractor
            sentence_distractor_alt = generate_alternative_sentence(sentence_distractor)
            sample["sentence_distractor_alt"] = sentence_distractor_alt
            counterexample["sentence_target_alt"] = sentence_distractor_alt

            distinct_sentences.add(sentence_target)
            distinct_sentences.add(sentence_distractor)
            distinct_sentences.add(sentence_target_alt)
            distinct_sentences.add(sentence_distractor_alt)

    print("Generated unique sentences:")
    for sentence in distinct_sentences:
        print(sentence)

    with open(args.eval_set, 'w') as file:
        json.dump(eval_set, file)


def get_common_predicate(predicates_target, predicates_distractor):
    for p in predicates_target:
        for p_d in predicates_distractor:
            if p == p_d:
                return p
    raise RuntimeError(
        f"No common predicates found between {predicates_target, predicates_distractor}"
    )


def get_predicates(captions, nlp, subject, predicate, obj, original_object):
    if original_object in OBJECTS_VERBS:
        return ["is"]

    else:
        if "(made of)" in obj:
            return ["is made of"]
        elif original_object == "Football":
            return [predicate + "s"]
        elif original_object == "Chair":
            return [predicate]

        # In these cases we just need to change the verbs to present progressive:
        elif original_object == "Dog":
            return ["holding"]
        elif original_object == "Wine glass" or original_object == "Mobile phone":
            return ["holding"]
        elif original_object == "Bicycle helmet":
            return ["wearing"]
        elif original_object == "Balloon":
            return ["holding"]
        elif original_object == "Wheelchair":
            return ["riding"]
        elif original_object == "Beer":
            return ["holding"]
        elif original_object == "Crown":
            return ["wearing"]
        elif original_object == "Cake":
            return ["holding"]
        else:
            # For ambiguous predicates: find most frequent usage in training data
            connections = []
            for caption in captions:
                tokens = caption.lower().split(" ")
                if subject in tokens and obj in tokens:
                    parsed = nlp(caption)
                    token_subject = [t for t in parsed if t.text == subject][0]
                    token_object = [t for t in parsed if t.text == obj][0]

                    if token_object.head == token_subject:
                        connections.append("")
                    elif token_subject.head == token_object.head:
                        if token_subject.head == token_object:
                            continue
                        if len(token_subject.head.conjuncts) > 0:
                            continue
                        else:
                            connections.append(token_subject.head.text)
                    else:
                        if token_object.head.head == token_subject:
                            if token_object.head.dep_ not in ["prep", "conj"]:
                                connections.append(token_object.head.text)

            assert len(connections) > 0, f"No connections found for {subject, obj}"
            conn_counter = Counter(connections)
            return [p for p, count in conn_counter.most_common()]


def generate_sentence_from_triplet(subj, pred, obj, original_object):
    if original_object not in OBJECTS_VERBS + OBJECTS_PLURAL + OBJECTS_TEXTURES:
        if not (obj == "guitar" or obj == "cello"):
            obj = "a " + obj

    if original_object in OBJECTS_TEXTURES:
        sentence = f"a {obj} {subj} {pred}".strip()
    else:
        sentence = f"a {subj} {pred} {obj}"

    # Lower case
    sentence = sentence.lower()

    return sentence


def generate_alternative_sentence(sentence):
    sentence_alt = sentence
    sentence_alt = "the" + sentence_alt[1:]
    sentence_alt = sentence_alt.replace(" is ", " ")
    sentence_alt = sentence_alt.replace("ting", "s")
    sentence_alt = sentence_alt.replace("ning", "s")
    sentence_alt = sentence_alt.replace("lding", "lds")
    sentence_alt = sentence_alt.replace("iding", "ides")
    sentence_alt = sentence_alt.replace("nding", "nds")
    sentence_alt = sentence_alt.replace("ading", "ads")
    sentence_alt = sentence_alt.replace("ling", "les")
    sentence_alt = sentence_alt.replace("ging", "gs")
    sentence_alt = sentence_alt.replace("king", "ks")
    sentence_alt = sentence_alt.replace("aying", "ays")
    sentence_alt = sentence_alt.replace("rying", "ries")
    sentence_alt = sentence_alt.replace("ring", "rs")
    sentence_alt = sentence_alt.replace("ping", "ps")

    return sentence_alt


def simplify_noun(noun):
    if noun == "Sun hat":
        noun = "hat"
    elif noun == "Bicycle":
        noun = "bike"
    elif noun == "Bicycle helmet":
        noun = "helmet"
    return noun


def transform_label(label):
    if label in OBJECTS_VERBS:
        if label.endswith("t"):
            label += "ting"
        elif label.endswith("e"):
            label = label[:-1] + "ing"
        elif label.endswith("n"):
            label += "ning"
        else:
            label += "ing"

    else:
        label = simplify_noun(label)

        if "(made of)" in label:
            label = label.replace("(made of)", "")

    return label.lower()


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--eval-set",
        type=str,
        required=True,
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    generate_sentence_for_eval_set(args)
