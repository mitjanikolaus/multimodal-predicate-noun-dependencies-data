import argparse
import os
import pickle

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, VisualBertForPreTraining

from eval_model import eval_2afc


def get_classification_score(model, tokenizer, test_sentence, visual_features, correct_sentence):
    # TODO: first sentence should not influence 2AFC outcome!
    inputs = tokenizer(correct_sentence, test_sentence, return_tensors="pt")

    decoded_sequence = tokenizer.decode(inputs["input_ids"][0])
    print(decoded_sequence)

    visual_embeds = visual_features.unsqueeze(0)

    # TODO: verify visual token type IDs
    visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

    inputs.update({
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask
    })

    outputs = model(**inputs)

    # Labels for computing the sentence-image prediction (classification) loss. Input should be a sequence
    # pair (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:
    #
    # - 0 indicates sequence B is a matching pair of sequence A for the given image,
    # - 1 indicates sequence B is a random sequence w.r.t A for the given image.
    # Prediction scores of the sentence-image prediction (classification) head (scores of True/False continuation before SoftMax)
    seq_relationship_logits = outputs.seq_relationship_logits

    # Apply softmax and return probability for match:
    softmaxed = F.softmax(seq_relationship_logits[0], dim=0)
    return softmaxed[0].data


def load_image_features(path):
    img_features = pickle.load(open(path, "rb"))
    img_features = {os.path.basename(key): value for key, value in img_features.items()}
    return img_features


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=str, required=True)
    parser.add_argument("--img-features-path", type=str, required=True)

    parser.add_argument("--offline", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.offline:
        tokenizer = BertTokenizer.from_pretrained(os.path.expanduser("~/data/transformers/bert-base-uncased"))
        model = VisualBertForPreTraining.from_pretrained(os.path.expanduser("~/data/transformers/uclanlp/visualbert-nlvr2-coco-pre"))
    else:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.save_pretrained(os.path.expanduser("~/data/transformers/bert-base-uncased"))

        model = VisualBertForPreTraining.from_pretrained("uclanlp/visualbert-nlvr2-coco-pre")
        model.save_pretrained(os.path.expanduser("~/data/transformers/uclanlp/visualbert-nlvr2-coco-pre"))

    img_features = load_image_features(args.img_features_path)

    eval_2afc(model, tokenizer, img_features, get_classification_score, args)

